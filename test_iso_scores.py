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
    points = np.transpose(points)	
    pca = PCA(n_components=len(np.transpose(points)))
    points = pca.fit_transform(points)	
    return np.transpose(points)
    
def gauss_line(dim, num_gauss, num_line):
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

#################################################################
#################################################################
#################################################################
###########   Existing Scores in the Literature  
#################################################################
#################################################################
#################################################################
def cosine_score(points, num_samples=1000, pca_reconfig=False):
    if pca_reconfig == True:
        points = pca_normalization(points)		
    points = points.T		
    cos_sim = []
    for _ in range(num_samples):
        p1 = np.reshape(points[np.random.randint(len(points))],(1,-1))
        p2 = np.reshape(points[np.random.randint(len(points))],(1,-1))
        #if np.array_equal(p1,p2):
            #continue 	
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
    n = np.shape(points)[0]
    pca_model = PCA(n_components=n)
    pc_embed = pca_model.fit_transform(points.T))
    var_explained = pca_model.explained_variance_ratio_.cumsum()
    num_components = int(np.floor(n*p))
    uniform_pc = num_components = num_components/n
    scores = uniform_pc/var_explained[num_components]
    return scores 

def id_score(points):
    n = np.shape(points)[0]
    lid_model = MLE()
    dim_est = lid_model.fit(points.T).dimension_
    score = dim_est/n
    if score > 1: return 1
    else: return score
   
#################################################################
#################################################################
###########   Testing Suite 
#################################################################
#################################################################
#################################################################


def gaussian_tests(score): 
    
    print("GAUSSIAN TESTS FOR", score.__name__)
    print("----------------------------------------------------------------------------")	
     
    print("TEST 1: Approaches 1 as we increase sample size.") 			
    smol_sample = sample_gaussian(num_points=100, dim=3, cov=1, mean=0)   
    print("Isotropic 100 Point Sample:", score(smol_sample))	
    big_sample = sample_gaussian(num_points=10000, dim=3, cov=1, mean=0)
    print("Isotropic 10,000 Point Sample:", score(big_sample))
         
    print("TEST 2: invariant to changes in dimension") 
    print("Dim = 1000")	
    big_dim = sample_gaussian(num_points=10000, dim=1000, cov=1, mean=0)	
    print("Isotropic Big Dim:", score(big_dim)) 
    
    print("TEST 3: invariant to the mean") 
    print("Mean = 1000")	
    big_mean = sample_gaussian(num_points=1000, dim=10, cov=1, mean=1000)	
    print("TEST 3 Isotropic Big Mean:", score(big_mean))  

    print("TEST 4: invariant to scalar covariance")
    print("Big Cov = 1000*I_n")	
    big_cov = sample_gaussian(num_points=1000, dim=10, cov=1000, mean=0)	
    print("Isotropic Big Cov:", score(big_cov)) 	
    print("Covariance Now 1e-5*I_n")	
    very_smol_cov = sample_gaussian(num_points=1000, dim=10, cov=1e-5, mean=0)	
    print("Very Small Cov:", score(very_smol_cov))	

    print("TEST 5: full cov matrix. Anisotropic, should hit lower bound of the score") 
    cov = np.full((10,10),1) 	
    full_cov = sample_gaussian(num_points=1000, dim=10, cov=cov, mean=0, uniform_cov=False)	
    print("Guassian Full Cov:", score(full_cov)) 	
        
    print("TEST 6: Unbalanced Cov") 
    cov = np.array([[100000,0,0],[0,1e-10,0],[0,0,1e-10]]) 
    print("Unbalaned Covariance")
    print(cov)	
    wonky_cov = sample_gaussian(num_points=1000, dim=3, cov=cov, mean=0, uniform_cov=False)	
    print("Guassian Unbalanced Cov:", score(wonky_cov)) 	 	
    print("----------------------------------------------------------------------------")	
    return None 

def hyperplane_tests(score):
    print("LINE AND HYPERPLANE TESTS FOR", score.__name__)
    print("----------------------------------------------------------------------------")	
    # TEST 8: perfectly anisotropic space; points live along a line 
    # dimension = 3, just sample from first axis
    cov = np.array([[100,0,0],[0,1e-5,0],[0,0,1e-5]])
    points = sample_gaussian(num_points=1000, dim=3, cov=cov, mean=0, uniform_cov=False)
    print("Anisotropic: points sampled just along x-axis in 3-space:", score(points))
    # dimension = 1000, just sample from first axis, with small noise in other axis
    cov = np.eye(1000)
    cov[0][0]=100000000
    points = sample_gaussian(num_points=10000, dim=1000, cov=cov, mean=0, uniform_cov=False)
    print("Anisotropic: points sampled just along x-axis in 1000-space:", score(points))
    # dimension = 3, just have points along line x=y=z
    cov = np.array([[10000,0,0],[0,1e-5,0],[0,0,1e-5]])
    points = sample_gaussian(num_points=10000, dim=3, cov=cov, mean=0, uniform_cov=False)
    points[1], points[2] = points[0], points[0]
    #print(np.cov(points))	
    print("Anisotropic: points sampled just along line x=y=z in 3-space:", score(points))
    # dimension = 1000, just sample from line where all coordinates are equal
    cov = np.eye(1000)
    cov[0][0]=100000000
    points = sample_gaussian(num_points=100000, dim=1000, cov=cov, mean=0, uniform_cov=False)
    for i in range(999):
        points[i+1] = points[0]
    print("Anisotropic: points sampled just along line where all coordinates are equal in 1000-space:", score(points))
    # dimension = 3, just have points along line x,y,z=(t,2t,-3t)
    cov = np.array([[10000,0,0],[0,1e-5,0],[0,0,1e-5]])
    points = sample_gaussian(num_points=10000, dim=3, cov=cov, mean=0, uniform_cov=False)
    points[1], points[2] = 2*points[0], -3*points[0]
    #print("First point in this next data set:", points[0][0],points[1][0],points[2][0])
    print("Anisotropic: points sampled just along line x,y,z=(t,2t,-3t) in 3-space:", score(points))

    # TEST 9: semi-anisotropic space; points live alone affine plane.	
    # dimension = 3, just have points sampled along the plane x+2y-3z=4
    dim = 1000	
    cov = np.array([[dim,0,0],[0,dim,0],[0,0,0]])
    points = sample_gaussian(num_points=dim, dim=3, cov=cov, mean=0, uniform_cov=False)
    for i in range(dim):
        points[2][i] = (4 - points[0][i] - 2*points[1][i])/(-3)
    print("Semi-anisotropic: points sampled along affine plane x+2y-3z=4 in 3-space:", score(points))
    #print("Semi-anisotropic PCA:", pca_var(points.T,comps=3,num_var=0)) 
    
    # dimension = 3, just have points sampled along the plane x-5y-z=-6
    cov = np.array([[dim,0,0],[0,dim,0],[0,0,0]])
    points = sample_gaussian(num_points=dim, dim=3, cov=cov, mean=0, uniform_cov=False)
    for i in range(dim):
        points[2][i] = (6 + points[0][i] - 5*points[1][i])
    print("Semi-anisotropic: points sampled along affine plane x-5y-z=-6 in 3-space:", score(points))
    #print("Semi-anisotropic PCA:", pca_var(points.T,comps=3,num_var=0))

    print("----------------------------------------------------------------------------")	
    return None 

def dim_used_test(dim,score):
    print("------------------------------------------------------------------------------------------")
    print("INCREASING DIMENSIONS USED TEST FOR",score.__name__)
    print("------------------------------------------------------------------------------------------")
    covariance_matrix = np.zeros((dim,dim))
    y = []
    for this_dim in range(dim):
        # change the (dim, dim) entry of this covariance matrix to one; compute covariance of sample points
        covariance_matrix[this_dim][this_dim] = 1
        sample_points = sample_gaussian(num_points=1000, dim=dim, cov=covariance_matrix, mean=0, uniform_cov=False)
    result = score(sample_points)
    print("Score when {} out of {} dimensions are utilized: {}".format(this_dim+1,dim,result))
    y.append(result)
    return np.array(y)

def max_var_test(score, dim, cov_range, mean=0):
    print("------------------------------------------------------------------------------------------")
    print("INCREASING MAX VARIANCE TEST FOR",score.__name__)
    print("------------------------------------------------------------------------------------------")
    score_scale = []		
    for x in range(1,cov_range):
        cov = np.eye(dim)
        cov[0][0] = x
        sample = sample_gaussian(num_points=1000, dim=dim, cov=cov, mean=mean, uniform_cov=False)
        result = score(sample)
    score_scale.append(result)
    print("Score of {} when max variance is {}".format(result,x))
    return score_scale

# execute this function to play with PCA normalization
def PCA_normalization_testing():

    # GENERATE POINTS: dimension = 3, just have points along line x=y=z
    pre_cov = np.array([[10000,0,0],[0,1e-5,0],[0,0,1e-5]])
    points = sample_gaussian(num_points=10000, dim=3, cov=pre_cov, mean=0, uniform_cov=False)
    points[1], points[2] = points[0], points[0]
    cov = np.cov(points)
    print("Covariance before normalization: \n", cov)


    # Normalizes points so that the principal components align with Euclidina axes
    print("Shape of points before PCA normalization: ", points.shape)
    points = PCA_normalization2(points)
    print("Shape of points after PCA normalization: ", points.shape)
    cov = np.cov(points)
    print("Covariance after normalization: \n", cov.shape, cov)

    return None 

def check_that_IsoScore_to_Dimension_correspondence_make_sense(n):
    print("------------------------------------------------------------------------------------------")
    print("check_that_IsoScore_to_Dimension_correspondence_make_sense for dim = {}".format(n))
    print("------------------------------------------------------------------------------------------")
    print("map_k_to_Iso_Score:")
    for k in range(1,n+1):
        print("  dim = {} maps to iota = {}".format(k, map_k_to_Iso_Score(n,k)))
    print("map_Iso_Score_to_k:")
    iota = 0.0
    while iota <= 1.0:
        print("  iota = {} maps to dim = {}".format(iota,map_Iso_Score_to_k(iota,n)))
        iota += 0.05

    return None


def increasing_dimensions_used(n, the_score):
    print("----------------------------------------------------------------------------")
    cov = np.full((n,n),0)
    for i in range(n):
        cov[i][i] = 1
        points = sample_gaussian(num_points=100000, dim=n, cov=cov, mean=0, uniform_cov=False)
        score = the_score(points)
        print("{} for points sampled using I_{}^{}: {}".format(the_score.__name__,n,i+1,score))
    print("----------------------------------------------------------------------------")
    return None

def half_of_dimensions_used_test(max_dim, score):

    print("----------------------------------------------------------------------------")
    print("Computing {} when half of available dimensions are utilized: ".format(score.__name__))
    print("----------------------------------------------------------------------------")
    for dim in range(2, max_dim+1, 2):
        cov = np.zeros((dim,dim))
        for i in range(dim//2):
            cov[i][i] = 1
        #points = sample_gaussian(num_points=10000, dim=dim, cov=cov, mean=0, uniform_cov=False)
        #the_score = score(points)
        the_score = iso_score_of_cov_matrix(cov)
        print("I_{}^{} -> {}, dims utilized = {}".format(dim,dim//2,the_score,map_Iso_Score_to_k(the_score,dim)))
    
    return None

def only_first_dimension_used_test(max_dim, score):
    print("----------------------------------------------------------------------------")
    print("Computing {} when only first dimensions is utilized: ".format(score.__name__))
    print("----------------------------------------------------------------------------")
    for dim in range(2, max_dim+1):
        cov = np.zeros((dim,dim))
        cov[0][0] = 1
        points = sample_gaussian(num_points=10000, dim=dim, cov=cov, mean=0, uniform_cov=False)
        the_score = score(points)
        print("I_{}^1: {}, dims utilized: {}".format(dim,the_score,map_Iso_Score_to_k(the_score,dim)))
    
    return None 

#################################################################
#################################################################
#################################################################
###########   Running the tests
#################################################################
#################################################################
#################################################################



def main():
    
    gaussian_tests(iso_score)	
    #print()
    hyperplane_tests(iso_score)

    run_ID_benchmarks()
    dim = 40
    #increasing_dimensions_used(dim, iso_score)
    half_of_dimensions_used_test(dim, iso_score)
    only_first_dimension_used_test(dim, iso_score)
    #test_conjecture(100000)


if __name__ == '__main__':
    main()




