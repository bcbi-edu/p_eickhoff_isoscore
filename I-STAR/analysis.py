import numpy as np
import hickle 
from sklearn.decomposition import PCA
from sklearn import manifold, datasets
from skdim.id import *
from skdim.datasets import *
import skdim
import torch
from training_utils import * #squad_eval, classification_eval, get_ci, load_classification_objs  
import glob
from regularizers import istar, cosine_regularizer 
from torch.utils.data import DataLoader 
from IsoScore import IsoScore

def id_estimate(id_model, data):
    """
    INPUT: 
        id_model: built-in skdim.id estimator. We exclusivel use TwoNN for this paper. 
        data: point cloud of data. In this paper, we look at LLM representations.  
    OUTPUT:
        The intrinsic dimension estimate of the data. 
    """
    # Note: this function only works for the built-in skdim.id estimators
    return id_model().fit(data).dimension_

def collate_fn(batch):
    """
    INPUT:
        batch: mini-batch of data used for stochastic gradient descent. 
    OUTPUT:
        outputs: A dictionary of input ids, attention masks and labels for the data. We use this in our DataLoader.
    """  
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

def run_analysis(config, model, train_data, eval_data, raw_data=None, squad_val=None, layer_id=False, save_states=False):  
    """
    INPUT: 
        config: argparse config containing the model name, task, random seed and more!
        model: a fine-tuned LLM model.
        train_data: data used to train the LLM. We use it here to compute the shrinkage covariance matrix for IsoScore*
        eval_data: test data used to evaluate the performance of the model. We compute all statistics (ID, IsoScore*, cosine similarity, mean/std of last layer sentence embeddings) of the given LLM model on the eval data. 
        raw_data/squad_val: used to run squad_eval
        layer_id: If TRUE we compute the ID of sentence embeddings of each layer in the model. Default is FALSE, bc it can be expensive (expect 1-2 hours depending on the model)! 
        save_states: If TRUE, we can save the sentence and predictions of the fine-tuned model on the validation data. Didn't do this in the paper, but could be a fun to do analysis on. 
        
    OUTPUT: 
        results: python dictionary containing the following stats on the model representations.
            1. Performance on the task
            2. IsoScore* of model representations
            3. ID of all model representations
            4. Cosine Similarity of last layer representations
            4. Mean/STD of last layer representations
            5. (optional) intrinsic dimensionality of representations from each FF hidden state in the model.
        results will be saved as a hickle file. 
    """
    
    print(config) 
    # Specify the last layer of the model     
    s = istar()  
    results = {}   
    if config.task == "squad": 
        performance, token_states, sentence_states, preds = squad_eval(config, eval_data, squad_val, raw_data, model, save_states=True, sentence_embed=True) 
    else:
        performance, token_states, sentence_states, preds = classification_eval(config, eval_data, model, save_states=True, sentence_embed=True) 
  
    # SAVE SENTENCE EMBEDDINGS... if you want. 
    if save_states:
        hickle.dump(sentence_states, config.model_name + "_" + config.task + "_" + config.seed + "_states.hickle", mode=w)
        hickle.dump(preds, config.model_name + "_" + config.task + "_" + config.seed + "_preds.hickle", mode=w)
    
    # Store performance  
    results["performance"] = performance 
    # Stack all states for a sigle vector space of all representations
    stack = []
    for layer in token_states.keys():
        if layer == "labels":
            continue 
        stack.append(np.vstack(token_states[layer]))  
    stack_states = np.vstack(stack).T 
    id_model = TwoNN
    results["all_isotropy"] = IsoScore.IsoScore(stack_states)  
    results["all_id"] = id_estimate(id_model,stack_states) 
    # Compute IsoScore* and mean vector for every layer in the model  
    results["layer_isotropy"] = [] 
    if layer_id: 
        results["layer_id"] = [] 
    for layer in sentence_states.keys(): 
        if layer == "labels":
            continue         
        config.layer = layer 
        ci = get_ci(config, train_data, model, max_points=250000) 
        # NOTE this is for sentence embeddings. May be interesting to look at token embeddings as well. 
        points=np.vstack(sentence_states[layer])
        points=torch.tensor(points) 
        score,_ = s.isoscore_star(points, ci, zeta=0.75, is_eval=True)
        results["layer_isotropy"].append(score.item())    
        # Compute ID of the sentence representations if true. This can take some time. 
        if layer_id:
            results["layer_id"].append(id_estimate(id_model,points)) 
        # Compute mean and avg cosine if we are on the last layer of the model 
        if layer == 12:
            points=np.vstack(token_states[layer])
            points=torch.tensor(points)  
            # Calculate cosine sim, mean and std here 
            cos_sim = cosine_regularizer()
            results["cos_sim"] = cos_sim.forward(points) 
            results["mean"] = torch.mean(points, axis=0) 
            results["std"] = torch.std(points, axis=0)  
    hickle.dump(results, config.model_name + "_" + config.regularizer + "_"+ str(config.tuning_param) + "_" +  str(config.seed) + "_" + config.task + "_" + str(config.layer) + "_final.hickle", mode='w')  
    return results

