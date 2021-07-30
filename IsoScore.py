# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA


#################################################################
#################################################################
#################################################################
###########   Helper Functions
#################################################################
#################################################################
#################################################################
	 

# Step 2
def pca_normalization(points):

    points = np.transpose(points)	
    pca = PCA(n_components=len(np.transpose(points)))
    points = pca.fit_transform(points)

    return np.transpose(points)

# Step 3
def get_diag_of_cov(points):

    n = np.shape(points)[0]
    cov = np.cov(points)
    cov_diag = cov[np.diag_indices(n)]

    return cov_diag

# Step 4
def normalize_diagonal(cov_diag):

    n = len(cov_diag)
    cov_diag_normalized = (cov_diag*np.sqrt(n))/np.linalg.norm(cov_diag)

    return cov_diag_normalized

# Step 5
def get_isotropy_defect(cov_diag_normalized):

    n = len(cov_diag_normalized)
    iso_diag = np.eye(n)[np.diag_indices(n)]	
    l2_norm = np.linalg.norm(cov_diag_normalized - iso_diag)	
    normalization_constant = np.sqrt(2*(n-np.sqrt(n)))
    isotropy_defect = l2_norm/normalization_constant

    return isotropy_defect

# Steps 6 and 7
def get_IsoScore(isotropy_defect, points):

    n = np.shape(points)[0]
    the_score = ((n-(isotropy_defect**2)*(n-np.sqrt(n)))**2-n)/(n*(n-1))

    return the_score


#################################################################
#################################################################
#################################################################
###########   Definition of IsoScore
#################################################################
#################################################################
#################################################################

def IsoScore(points):

    # Step 2
    points_PCA = pca_normalization(points)

    # Step 3
    cov_diag = get_diag_of_cov(points_PCA)

    # Step 4
    cov_diag_normalized = normalize_diagonal(cov_diag)

    # Step 5
    isotropy_defect = get_isotropy_defect(cov_diag_normalized)

    # Steps 6 and 7
    the_score = get_IsoScore(isotropy_defect, points)

    return the_score