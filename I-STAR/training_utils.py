#PyTorch & Numpy
import torch 
from torch.utils.data import DataLoader 
import numpy as np


# Custom GPT2 for QA model:
from gpt2_qa import GPT2ForQuestionAnswering

#HuggingFace Stuff 
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2TokenizerFast, GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2ForSequenceClassification
from transformers import BertTokenizerFast, BertModel, BertConfig, BertForSequenceClassification,  BertForQuestionAnswering
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertConfig, DistilBertForSequenceClassification,  DistilBertForQuestionAnswering
from transformers import AlbertTokenizerFast, AlbertModel, AlbertConfig, AlbertForSequenceClassification, AlbertForQuestionAnswering
from transformers import AdamW, get_scheduler
from datasets import load_dataset, load_metric  
from transformers import default_data_collator
import evaluate
import collections

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

# Custom Functions 
from nc0_regularizers import istar, cosine_regularizer
from analysis import * 

def get_ci(config, data, model, max_points, gpu_id=None):
    """Given the data and model of interest, generate a sample of size max_points,
    then calculate the covariance matrix. Run this as a warmup to generate a stable
    covariance matrix for IsoScore Regularization"""       
    # Send model to gpu if available
    if gpu_id: 
        device = gpu_id 
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'        
    num_points = 0 
    points_list = []
    model.eval() 
    # main EVAL loop 
    for idx, batch in enumerate(data):
        # send batch to device  
        batch = {key: value.to(device) for key, value in batch.items()} 
       
        # Set model to eval and run input batches with no_grad to disable gradient calculations   
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True) 
            if config.layer == "all": 
                points = torch.reshape(torch.stack(outputs.hidden_states)[1:,:,:,:], (-1,768))
            else: 
                points = torch.reshape(outputs.hidden_states[int(config.layer)], (-1,768))
        num_points += points.shape[0]
       # Collect the last state representations to a list and keep track of the number of points    
        points = points.detach().cpu().numpy()  
        points_list.append(points)   
        if num_points > max_points:
            break
    # Convert model back to train mode: 
    model.train()
    # Stack the points and calclate the sample covariance C0 
    sample = np.vstack(points_list)
    C0 = np.cov(sample.T)
    return torch.tensor(C0)

# NOTE: Need to turn off Distributed sampling if you're only using a single GPU 
def prepare_dataloader(config, dataset, is_eval=False): 
    if config.task == "squad":
        collate_fn = default_data_collator    
    else:
        collate_fn = classification_collate_fn
    
    if is_eval:   
        if config.task == "squad": 
            dataset = dataset.remove_columns(["example_id", "offset_mapping"])
            dataset.set_format("torch")         
        dl = DataLoader(
                dataset,
                batch_size = config.batch_size, 
                pin_memory=True, 
                shuffle=False,
                collate_fn=collate_fn, 
                num_workers=4)  
    else:  
        if config.task == "squad": 
            dataset.set_format("torch")   
            sampler=DistributedSampler(dataset),
        dl = DataLoader(
                dataset,
                batch_size = config.batch_size, 
                pin_memory=True, 
                sampler=DistributedSampler(dataset),
                shuffle=False,
                collate_fn=collate_fn, 
                num_workers=4)    
    return dl 

def sow_seeds(seed):
    #Sow seeds 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    return None

########################### Classification Specific Functions ###########################
def classification_collate_fn(batch):
    """ Collate function used to make batches for the DataLoader"""  
    max_len = max([len(f["input_ids"]) for f in batch])  
    #Pad examples in the batch to be the same len 
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    # Retrieve labels 
    labels = [f["labels"] for f in batch]
    # Tensors need to be floats, labels need to long 
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    # Make the batch into a nice dictionary of input_ids, the attention mask and labels    
    outputs = { "input_ids": input_ids, "attention_mask": input_mask, "labels": labels }
    return outputs


def classification_eval(config, eval_loader, model, save_states=False, sentence_embed=False):
    # Set model to eval mode. Load metric and create data loader.  
    if save_states:
        print("SAVE STATES")
    if sentence_embed:
        print("SENTENCE EMBED") 
    model.eval() 
    num_saved_points = 0
    # Send model to gpu if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # Lists to store results. 
    preds_list = []
    labels_list = [] 
    states_list = {"labels": []} 
    sentence_list = {}  
    
    if config.model_name == "distbert":
        num_layers=7
    else:
        num_layers=13 
    
    for i in range(1,num_layers):
        states_list[i] = []
        sentence_list[i] = [] 
    
    for idx, batch in enumerate(eval_loader):
        # send batch to device  
        batch = {key: value.to(device) for key, value in batch.items()}  
        # Set model to eval and run input batches with no_grad to disable gradient calculations   
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=save_states) 
            logits = outputs.logits   
        # CAN add a statement to limit the number of points saved 
        if save_states:
            if num_saved_points > 250000:
                continue 
            states = outputs.hidden_states 
            if config.model_name in ["bert", "distbert", "albert"]:
                #for i in range(1,13,1):       
                 for i in range(1, len(states)):   
                     # ALL TOKEN EMBEDDINGS.  
                     states_list[i].append(torch.reshape(states[i], (-1,768)).detach().cpu().numpy())      
                     # SENTENCE EMBEDDINGS 
                     if sentence_embed==True: 
                        sentence_list[i].append(torch.reshape(states[i][:,0,:], (-1,768)).detach().cpu().numpy()) 
            if config.model_name == "gpt2":
                 for i in range(1, len(states)):       
                     # ALL TOKEN EMBEDDINGS 
                     states_list[i].append(torch.reshape(states[i], (-1,768)).detach().cpu().numpy()) 
                     # SENTENCE EMBEDDINGS 
                     if sentence_embed==True: 
                        sentence_list[i].append(torch.reshape(states[i][:,-1,:],(-1,768)).detach().cpu().numpy())    
            num_saved_points += np.vstack(states_list[i]).shape[0] 
        # Store Predictions and Labels
        preds = logits.argmax(axis=1)        
        preds = preds.detach().cpu().numpy()  
        preds_list.append(preds)   
        labels = batch["labels"].detach().cpu().numpy() 
        states_list["labels"].append(labels) 
        labels_list.append(labels)  
    # Compute Accuracy 
    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    acc = (preds == labels).sum()/len(preds)
    # Set model to train!  
    model.train() 
    # Return acc, points states and sentence states 
    if save_states==True and sentence_embed==True:
        return acc, states_list, sentence_list, preds 
    elif save_states:
        return acc, states_list, preds 
    else:
        return acc

def load_classification_objs(config):
    """ 
    1) loads the specified dataset then preprocesses/tokenizes both the train and the eval data so it can easily be fed into a DataLoader. 
    2) load model specified in config.  
    3) load the optimizer. 
    """ 
    # Preprocessing and tokenizing data
    task_keys = {
            "sst2": ("sentence", None),
            "sst": ("sentence", None), 
            "qnli": ("question", "sentence"),
            }   
     
    # Loading data for the given task 
    if config.task == "sst":
        data = load_dataset(config.task)
        num_labels = 5  
    else:
        data = load_dataset("glue", config.task)
        num_labels = len(data["train"].features["label"].names)

    # Loading the specified model AND tokenizer 
    if config.model_name == "gpt2": 
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") 
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token                            
        model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=num_labels) 
        model.config.pad_token_id = model.config.eos_token_id 
    
    if config.model_name == "bert":  
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased") 
        tokenizer.padding_side = "right"        
        model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels) 
         
    if config.model_name == "distbert":  
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased") 
        tokenizer.padding_side = "right"        
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=num_labels)
    
    if config.model_name == "albert":   
        tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2") 
        tokenizer.padding_side = "right"        
        model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels) 
    
    # Preprocessing and tokenizing data
    def classification_preprocess(example): 
        key1, key2 = task_keys[config.task] 
        if key2 is None: inputs = (example[key1],)
        else: 
            inputs = (example[key1], example[key2])

        results = tokenizer(*inputs, max_length=256, truncation=True, add_special_tokens=True) 
        if config.task == "sst":
            results["labels"] = np.digitize(example["label"], np.array([0.2,0.4,0.6,0.8]), right=True)
        else:
            results["labels"] = example["label"] #if "label" in example else 0  
        return results    
    
    # For debugging purposes, set train equal to "mini" 
    if config.training == "Mini": 
        train_data = list(map(classification_preprocess, data["train"]))[:10000] 
        eval_data = list(map(classification_preprocess, data["validation"])) 
    else: 
        train_data = list(map(classification_preprocess, data["train"])) 
        eval_data = list(map(classification_preprocess, data["validation"])) 
   
    optimizer =  AdamW(model.parameters(), lr=config.learning_rate) #load optimizer and learning rate. 
    return model, train_data, eval_data, optimizer

def classification_collate_fn(batch):
    """ Collate function used to make batches for the DataLoader"""  
    max_len = max([len(f["input_ids"]) for f in batch])  
    #Pad examples in the batch to be the same len 
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    # Retrieve labels 
    labels = [f["labels"] for f in batch]
    # Tensors need to be floats, labels need to long 
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    # Make the batch into a nice dictionary of input_ids, the attention mask and labels    
    outputs = { "input_ids": input_ids, "attention_mask": input_mask, "labels": labels }
    return outputs

########################### SQUAD Specific Functions ###########################
def compute_metrics(start_logits, end_logits, features, examples):
    n_best=20 
    max_answer_length=30 
    metric = load_metric("squad")
    
    example_to_features = collections.defaultdict(list)    
    for idx, feature in enumerate(features): 
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def squad_eval(config, eval_loader, eval_data, raw_data, model, save_states=False, sentence_embed=False):
    print("Evaluating") 
    model.eval()
    num_save_points = 0 
    
    print("dataloader made") 
    # Specify Metrics
    metric = evaluate.load("squad") 
    
    # Send model to gpu if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Lists to store results. 
    start_logits = []
    end_logits = [] 
        
    states_list = {"labels": []} 
    sentence_list = {}  
    
    if config.model_name == "distbert":
        num_layers=7
    else:
        num_layers=13 
    
    for i in range(1,num_layers):
        states_list[i] = []
        sentence_list[i] = []
    num_saved_points = 0
    # main EVAL loop  
    for idx, batch in enumerate(eval_loader):
        # send batch to device  
        batch = {key: value.to(device) for key, value in batch.items()} 
               
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=save_states) 
         
        start_logits.append(outputs.start_logits.detach().cpu().numpy())   
        end_logits.append(outputs.end_logits.detach().cpu().numpy())
        
        if save_states:
            if num_saved_points > 250000:
                continue 
            states = outputs.hidden_states 
            if config.model_name in ["bert", "distbert", "albert"]:
                #for i in range(1,13,1):       
                 for i in range(1, len(states)):   
                     # ALL TOKEN EMBEDDINGS.  
                     states_list[i].append(torch.reshape(states[i], (-1,768)).detach().cpu().numpy())      
                     # SENTENCE EMBEDDINGS 
                     if sentence_embed==True: 
                        sentence_list[i].append(torch.reshape(states[i][:,0,:], (-1,768)).detach().cpu().numpy()) 
            if config.model_name == "gpt2":
                 for i in range(1, len(states)):       
                     # ALL TOKEN EMBEDDINGS 
                     states_list[i].append(torch.reshape(states[i], (-1,768)).detach().cpu().numpy()) 
                     # SENTENCE EMBEDDINGS 
                     if sentence_embed==True: 
                        sentence_list[i].append(torch.reshape(states[i][:,-1,:],(-1,768)).detach().cpu().numpy())    
            num_saved_points += np.vstack(states_list[i]).shape[0] 
    
    print("batches processed") 
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits) 
    print("Trying compute metrics") 
    metrics = compute_metrics(start_logits, end_logits, eval_data, raw_data)
    
    # Set model to train!  
    model.train() 
    # Return EM/F1, token states and sentence states 
    if save_states==True and sentence_embed==True:
        return metrics, states_list, sentence_list 
    elif save_states:
        return metrics, states_list, (start_logits, end_logits) 
    else:
        return metrics
    
def load_squad_objs(config):
    """ 
    1) loads the specified dataset then preprocesses/tokenizes both the train and the eval data so it can easily be fed into a DataLoader. 
    2) load model specified in config.  
    3) load the optimizer. 
    """ 
    data = load_dataset("squad")
    print("Data loaded") 

    # Loading the specified model AND tokenizer 
    if config.model_name == "gpt2": 
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") 
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token                            
        model = GPT2ForQuestionAnswering.from_pretrained("gpt2")  
        model.config.pad_token_id = model.config.eos_token_id 
    
    if config.model_name == "bert":  
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased") 
        tokenizer.padding_side = "right"        
        model = BertForQuestionAnswering.from_pretrained("bert-base-cased") 
         
    if config.model_name == "distbert":  
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased") 
        tokenizer.padding_side = "right"        
        model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased")
    
    if config.model_name == "albert":   
        tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2") 
        tokenizer.padding_side = "right"        
        model = AlbertForQuestionAnswering.from_pretrained("albert-base-v2") 
    

    # Preprocessing Functions 
    max_length=384             
    stride=128 
    def preprocess_qa_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            )

        
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping): 
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context 
            if config.model_name=="gpt2":   
                idx = 0
                while sequence_ids[idx] == 0:
                    idx += 1
                context_start = idx

                while sequence_ids[idx] == 1:
                    idx += 1   
                    if idx == len(sequence_ids): 
                        idx -= 1 
                        break  
                context_end = idx-1 

            else:      
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx


                while sequence_ids[idx] == 1:
                    idx += 1    
                context_end = idx - 1 

            # If the answer is not fully inside the context, label is (0, 0)                                   
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char: 
                start_positions.append(0)
                end_positions.append(0) 
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx=context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_qa_val(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs            

    # Preprocess the data    
    train_data = data["train"].map(
        preprocess_qa_train, 
        batched=True,
        remove_columns=data["train"].column_names,
    )

    eval_data = data["validation"].map(
        preprocess_qa_val, 
        batched=True,
        remove_columns=data["validation"].column_names,
    )
    
    raw_data = data["validation"] 
    optimizer =  AdamW(model.parameters(), lr=config.learning_rate) #load optimizer and learning rate. 
    return model, train_data, eval_data, raw_data, optimizer



 
