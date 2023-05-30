import torch

class istar(torch.nn.Module): 
    """ 
    PyTorch module to implement I-STAR loss. Note that the isoscore_star function can be used to compute IsoScore* during evaluation and analysis of model representations.
    """
    def __init__(self): 
        super(istar, self).__init__() 
    
    def isoscore_star(self, points, C0, zeta, is_eval=False, gpu_id=None): 
        """
        INPUT: 
            points: point cloud of data that we use to compute IsoScore*
            C0: shrinkage covariance matrix. If you do not have access to a larger distribution of points, use the identity matrix to stabilize covariance matrix calculations!!! 
            zeta: shrinkage parameter used to control how much covariance matrix information is computed from C0 and the covariance matrix of the input points.
            is_eval: if TRUE we disable all gradient calculations. 
            gpu_id: variable to set device to a given gpu_id. Used in DDP. 
        
        OUTPUT: 
            The IsoScore* value of points. 
        """
        
        # make sure everything is on the same device!   
        if gpu_id:
            device = gpu_id
        else:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        # Make sure the input is a tensor and set requires_grad=True   
        if is_eval==True:
            points=torch.tensor(points)
        else: 
            points.requires_grad_() 
        # Getting the dimensionality of the point cloud  
        n = torch.tensor(points.shape[1])
        C = torch.cov(points.T)    
        cov = (1-zeta)*C + zeta*C0 
        # Compute singular values 
        pcs = torch.linalg.svdvals(cov) 
        # Normalize singular values 
        pcs_norm = (pcs*torch.sqrt(n))/torch.linalg.vector_norm(pcs) 
        # Normalization constant to bound the isotropy defect between [1/n, n]
        secret_spice = torch.sqrt(2*(n-torch.sqrt(n))) 
        # Compute distance between principal components and (1,1,...,1)
        if is_eval==True:
            ones = torch.ones(pcs.shape[0])
        else:
            ones = torch.ones(pcs.shape[0]).to(device) 
        l2_norm = torch.linalg.vector_norm(pcs_norm - ones)
        # Normalize    
        iso_defect = l2_norm/secret_spice 
        # Rescale
        score = ((n-(iso_defect**2)*(n-torch.sqrt(n)))**2-n)/(n*(n-1))         
        return score 

    def isoscore_cov(self, cov, gpu_id=None): 
        """ 
        INPUT: 
            cov: covariance matrix of a given distribution
        OUTPUT:
            IsoScore* for the covariance matrix. This is used when you already know the covariance matrix of the distribution 
            and do not need to use a point cloud to estimate the covariance matrix of a distribution. 
        """
        if gpu_id:
            device = gpu_id
        else:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  
        pcs = torch.linalg.svdvals(cov)
        #pcs.requires_grad_()
        n = torch.tensor(cov.shape[1]) 
        # Normalize singular values 
        pcs_norm = (pcs*torch.sqrt(n))/torch.linalg.vector_norm(pcs)
      
        # Normalization constant to bound the isotropy defect between [1/n, n]
        secret_spice = torch.sqrt(2*(n-torch.sqrt(n)))
        
        # Compute distance between principal components and (1,1,...,1)
        ones = torch.ones(pcs.shape[0]).to(device)  
        l2_norm = torch.linalg.vector_norm(pcs_norm - ones)   
        iso_defect = l2_norm/secret_spice
        
        # Rescale
        score = ((n-(iso_defect**2)*(n-torch.sqrt(n)))**2-n)/(n*(n-1)) 
        return score 
           
class cosine_regularizer(torch.nn.Module): 
    """ 
    PyTorch module implementing CosReg training as described by Gao et al. 2019
    https://arxiv.org/abs/1907.12009    
    """ 
    def __init__(self): 
        super(cosine_regularizer, self).__init__() 

    def forward(self,points):
        # Normalize points by norm. 
        norms = torch.reshape(1/torch.linalg.norm(points, dim=1),(-1,1))
        points = norms*points
        W = torch.matmul(points, points.T)
        N = W.shape[0] 
        return (torch.sum(W) - N)/N**2      
