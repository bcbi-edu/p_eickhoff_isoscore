# -*- coding: utf-8 -*-
from __future__ import unicode_literals


import numpy as np
from numpy import linalg
import random 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import uniform
import csv
import pandas as pd
import argparse
from sklearn.decomposition import PCA
import math
from numpy.random import multivariate_normal
from scipy.spatial import distance_matrix
import time 
from sklearn import datasets
from skdim.id import *

#################################################################
#################################################################
#################################################################
###########   Existing Scores in the Literature  
#################################################################
#################################################################
#################################################################
def cosine_score(points, num_samples=100000, pca_reconfig=False):
    """Given a point cloud, points, average random cosine of 100,000 tuples sampled from points.
    Please note that we report scores for 1 - abs(cosine_score(points)) for ease of comparison.""" 
    if pca_reconfig == True:
        points = pca_normalization(points)		
    points = points.T		
    cos_sim = []
    for _ in range(num_samples):
        p1 = np.reshape(points[np.random.randint(len(points))],(1,-1))
        p2 = np.reshape(points[np.random.randint(len(points))],(1,-1))	
        cos_sim.append(cosine_similarity(p1,p2))
    return sum(cos_sim)[0][0]/num_samples

def partition_score(points):
    """Partition score of isotropy taken from Mu et al. 2018.  
       1 is isotropic and 0 in anisotropic."""
    _, C = np.linalg.eig(np.matmul(points,points.T))
    scores = []
    for c in C:
        scores.append(np.sum(np.exp(c*points.T)))
    return min(scores)/max(scores) 

def varex_score(points, p=0.3):
    """Computes how uniform the first p% of princpal components of points are distributed """  
    n = np.shape(points)[0]
    pca_model = PCA(n_components=n)
    pc_embed = pca_model.fit_transform(points.T))
    var_explained = pca_model.explained_variance_ratio_.cumsum()
    num_components = int(np.floor(n*p))
    uniform_pc = num_components = num_components/n
    scores = uniform_pc/var_explained[num_components]
    return scores 

def id_score(points):
    """Computes the intrinsic dimensionality of points using MLE and divide by the true dimension of points"""  
    n = np.shape(points)[0]
    lid_model = MLE()
    dim_est = lid_model.fit(points.T).dimension_
    score = dim_est/n
    if score > 1: return 1
    else: return score
   

