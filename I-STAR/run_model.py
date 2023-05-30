#PyTorch & Numpy
import torch 
from torch.utils.data import DataLoader 
import numpy as np
# Pytorch stuff for DDP
from torch.utils.data.distributed import DistributedSampler 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.multiprocessing as mp
# Misc  
import hickle
from tqdm import tqdm 
import argparse 
import random 
import os
import wandb
import typing
from datasets import Dataset
from fvcore.nn import FlopCountAnalysis
# Custom Functions 
from regularizers import istar, cosine_regularizer
from analysis import * 
from training_utils import *

def ddp_setup(rank, world_size):
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        config: dict,  
        model: torch.nn.Module, 
        train_data: DataLoader, 
        eval_data: DataLoader, 
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        raw_data: typing.Optional[Dataset] = None, 
        squad_val: typing.Optional[Dataset] = None 
        ) -> None:  
        self.config = config  
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.eval_data = eval_data 
        self.optimizer = optimizer
        self.best_performance = 0
        self.model = DDP(model, device_ids=[gpu_id]) 
        self.scaler = torch.cuda.amp.GradScaler()
        self.raw_data = raw_data
        self.squad_val = squad_val
        # Specifying the regularization function in reg.  
        if self.config.regularizer == "cos_sim": 
            self.reg = cosine_regularizer() 
        if self.config.regularizer == "istar":
            self.reg = istar()       
     
    def _run_batch(self, batch, C0=None):
       
        """ 
        INPUT: source: mini-batch data, targets: mini-batch labels, C0: shrinkage covariance matrix (only for istar regularization)  
        This function computes a forward, then backward pass for a mini-batch of data. Choice of regularizer is specified in the config file.  
        """   
        
        self.optimizer.zero_grad() 
      
        # I-STAR Regularization  
        if self.config.regularizer == "istar": 
            with torch.cuda.amp.autocast(dtype=torch.float16): 
                outputs = self.model(**batch, output_hidden_states=True)
            # Cross entropy loss            
            output_loss = outputs.loss  
            if self.config.layer == "all":
                # stack outputs from all layers of the network, but NOT the initial embedding lookup.  
                points = torch.reshape(torch.stack(outputs.hidden_states)[1:,:,:,:], (-1,768))            
            else: 
                # get activations from the hidden layer we specify.  
                points = torch.reshape(outputs.hidden_states[int(self.config.layer)], (-1,768))                 
            # IsoScore* of points 
            batch_iso,_  = self.reg.isoscore_star(points, C0, zeta=self.config.zeta, gpu_id=self.gpu_id) 
            iso_score_loss = self.config.tuning_param*(1-batch_iso)   
            #I-STAR loss 
            loss = output_loss + iso_score_loss
       
        # Cosine-Similarity Regularization 
        if self.config.regularizer == "cos_sim":
            with torch.cuda.amp.autocast(dtype=torch.float16): 
                outputs = self.model(**batch, output_hidden_states=True)
            # Cross entropy loss            
            output_loss = outputs.loss  
            # In the paper, we just do layer 12. I keep all in here however since we tried this. But it hurt performance a lot.  
            if self.config.layer == "all":
                # stack outputs from all layers of the network.  
                points = torch.reshape(torch.stack(outputs.hidden_states)[1:,:,:,:], (-1,768))            
            else: 
                # get activations from the hidden layer we specify.  
                points = torch.reshape(outputs.hidden_states[int(self.config.layer)], (-1,768))  
            # Cosine similarity of points 
            cos_loss = self.reg.forward(points) 
            # Cos-Reg Loss  
            loss = output_loss + self.config.tuning_param*cos_loss 
        
        # No Regularizer.  
        if self.config.regularizer == "None": 
            with torch.cuda.amp.autocast(dtype=torch.float16): 
                outputs = self.model(**batch)    
            loss = outputs.loss 
        
        # Log loss every training step  
        if self.gpu_id == 0: 
            wandb.log({"Loss": loss})
       
       # Backprop (scaler for fp16 training) 
        self.scaler.scale(loss).backward()  
        self.scaler.step(self.optimizer) 
        self.scaler.update() 
        self.model.zero_grad()
    
    def _run_epoch(self, epoch): 
        """ 
        INPUT: the current epoch for a given gpu-id
        Sends mini-batches to device and calls _run_batch to complete a single epoch.
        Note: At the start of each epoch, we create a new shrinkage matrix to reflect changes in the models representations. 
        """ 
        b_sz = len(next(iter(self.train_data))['input_ids']) 
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        # Here we can call get_ci on gpu_id 1 and then broadcast it to other GPUs.  
        self.train_data.sampler.set_epoch(epoch)
        
        if self.config.regularizer == "istar":
            C0 = get_ci(self.config, self.train_data, self.model, max_points=250000, gpu_id=self.gpu_id) 
            C0 = C0.to(self.gpu_id)
        else:
            C0 = None
        
        # Send everything to device and call run_batch 
        for idx, batch in enumerate(self.train_data):
            batch = {key: value.to(self.gpu_id) for key, value in batch.items()}    
            self._run_batch(batch, C0)
   
    def _save_model(self):
        """ 
        Saves model checkpoint to PATH 
        """ 
        ckp = self.model.module.state_dict()
        PATH = "../models/" + self.config.model_name + "_" + self.config.regularizer + "_" + str(self.config.tuning_param) + "_" + str(self.config.seed) + "_" + self.config.task + "_" + str(self.config.layer) + ".pth" # HERE SAVE THE MODELS IF WE WANT? OTHERWISE SKIP
        torch.save(ckp, PATH)
        print("MODEL SAVED")
      
    def train(self): 
        """ 
        Train the model for num_epochs and save the model for the last epoch.  
        """ 
        wandb.watch 
        for epoch in range(self.config.num_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0: 
                if self.config.task == "squad":
                    results = squad_eval(config=self.config, eval_loader=self.eval_data, eval_data=self.squad_val, raw_data=self.raw_data, model=self.model)
                    em = results["exact_match"]
                    f1 = results["f1"] 
                    wandb.log({"EM": em, "F1": f1}) 
                    if em > self.best_performance: 
                        self._save_model() 
                        self.best_performance = em
                else:
                    acc = classification_eval(self.config, self.eval_data, self.model)  
                    wandb.log({"Accuracy": acc}) 
                        
                if epoch+1 == self.config.num_epochs: 
                    # SAVE MODEL AND RUN ANALYSIS 
                    run_analysis(config=self.config, model=self.model, train_data=self.train_data, eval_data=self.eval_data, raw_data=self.raw_data, squad_val=self.squad_val) 
                    self._save_model()               

def main(rank: int, config: dict, world_size: int):
    # Wandb init
    print(config) 
    # Monitor everything with wandb. NOTE: only logging metrics for GPU0. So, look at the results files and NOT these. This is just for monitoring experiments.  
    wandb.init(project=config.regularizer + "_" + config.task, name=config.model_name + "_" + str(config.zeta) + "_" + str(config.tuning_param) + "_" + str(config.seed))   
    # Train with DDP
    ddp_setup(rank,world_size) 
    # Sow seeds  
    sow_seeds(int(config.seed))
    if config.task == 'squad': 
        model, train_data, eval_data, raw_data, optimizer = load_squad_objs(config) 
    else:
        model, train_data, eval_data, optimizer = load_classification_objs(config)
      
    # Create dataloaders 
    train_loader = prepare_dataloader(config, train_data) 
    eval_loader = prepare_dataloader(config, eval_data, is_eval=True)  
        
    if config.task == "squad":
        trainer = Trainer(config, model, train_loader, eval_loader, optimizer, rank, raw_data=raw_data, squad_val=eval_data)
    else:
        trainer = Trainer(config, model, train_loader, eval_loader, optimizer, rank)
    
    trainer.train()
    destroy_process_group()

if __name__  == "__main__":
    # Argparser to create config 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=8, type=int) 
    parser.add_argument("--layer", default='all')
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--model_name", default="distbert", type=str)
    parser.add_argument("--seed", default=33, type=int) 
    parser.add_argument("--training", default="Mini", type=str) 
    parser.add_argument("--zeta", default=0.2, type=float) 
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--regularizer", default="istar", type=str)  
    parser.add_argument("--tuning_param", default=0.25, type=float)   
    config = parser.parse_args() 
    
    world_size = torch.cuda.device_count()
    print("DEVICE COUNT", world_size) 
    mp.spawn(main, args=(config, world_size), nprocs=world_size) 


