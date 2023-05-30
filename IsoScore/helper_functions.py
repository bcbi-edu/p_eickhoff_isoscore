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
###########   Helper Functions
#################################################################
#################################################################
########################

def sample_gaussian(num_points, dim, mean, cov, uniform_mean=True, uniform_cov=True): 
    """INPUTS:
       
       num_points: number of points in the sample. 
       dim: dimensionality of gaussian we are sampling from.
       mean: if uniform_mean==True then a scalar. Otherwise a dim dimensional vector of the mean.
       cov: if uniform_cov == True then a scalar to produce scaled identity for covariance. Otherwise custom dim x dim covariance matrix.
       
       RETURNS: 
       matrix of dim x num_points sampled from a multivariate_gaussian."""	
           
    # Create a unform mean and covariance:
    if uniform_mean == True:	
        mean = np.full((dim), mean)
    if uniform_cov == True:	
        cov = np.eye(dim)*cov
        
    # Use customized mean and covariance:	
    if uniform_mean == False:
        mean = mean	
    if uniform_cov == True:
        cov = cov	
    samples = multivariate_normal(mean=mean, cov=cov, size=num_points)	
    return samples.T 	 

def pca_normalization(points):
    """Projects points onto the directions of maximum variance.""" 
    points = np.transpose(points)	
    pca = PCA(n_components=len(np.transpose(points)))
    points = pca.fit_transform(points)	
    return np.transpose(points)
    
def skewered_meatball(dim, num_gauss, num_line):
    """Intersect points sampled from a multivariate Gaussian and a line in n dimensional space.""" 
    gauss = sample_gaussian(num_points=num_gauss, dim=dim, cov=1, mean=0)
    cov = np.full((dim,dim),1)
    line = sample_gaussian(num_points=num_line, dim=dim, mean=0, cov=cov, uniform_cov=False)
    points = np.hstack((line,gauss))
    return points

# computes closed-form expression for IsoScore of I_n^{(k)}
def map_k_to_Iso_Score(n, k):
    return 1-np.sqrt(n-np.sqrt(n*k)) / np.sqrt(n-np.sqrt(n))

# computes closed form expression for fxn which maps IsoScore to number of dimensions utilized
def map_Iso_Score_to_k(iota, n):
    return iota*(n-1) + 1


