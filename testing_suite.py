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
from helper_functions import *
from existing_scores import *
from IsoScore import *

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
        sample_points = sample_gaussian(num_points=100000, dim=dim, cov=covariance_matrix, mean=0, uniform_cov=False)
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
        sample = sample_gaussian(num_points=100000, dim=dim, cov=cov, mean=mean, uniform_cov=False)
        result = score(sample)
    score_scale.append(result)
    print("Score of {} when max variance is {}".format(result,x))
    return score_scale

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

    


if __name__ == '__main__':
    main()




