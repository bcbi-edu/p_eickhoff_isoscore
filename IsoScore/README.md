# IsoScore: Measuring the Uniformity of Embedding Space Utilization
 *by William Rudman, Nate Gillman, Taylor Rayne, and Carsten Eickhoff*

IsoScore is a tool that measures how uniformly a point cloud utilizes the Euclidian space that it sits inside of. 
See the original paper (https://arxiv.org/abs/2108.07344) appearing in the Findings of ACL 2022 for more information. 
This repository contains the Python3 implementation of IsoScore.

### How to install

The only dependencies are `numpy` and `sklearn`.

```
pip install IsoScore
```

### How to use

If you want to compute the IsoScore for a point cloud X that sits inside R^n, then X must be a `numpy` array of shape (n,m), where X contains m points, and the latent dimension is n.
For example:


```python3
import numpy as np
from IsoScore import IsoScore

random_array_1 = np.random.normal(size=100)
random_array_2 = np.random.normal(size=100)
random_array_3 = np.random.normal(size=100)

# Computing the IsoScore for points sampled from a line (dim=1) in R^3
point_cloud_line = np.array([random_array_1, np.zeros(100), np.zeros(100)])
the_score = IsoScore.IsoScore(point_cloud_line)
print(f"IsoScore for 100 points sampled from this line in R^3 is {the_score}.")

# Computing the IsoScore for points sampled from a disk (dim=2) in R^3
point_cloud_disk = np.array([random_array_1, random_array_2, np.zeros(100)])
the_score = IsoScore.IsoScore(point_cloud_disk)
print(f"IsoScore for 100 points sampled from this disk in R^3 is {the_score}.")

# Computing the IsoScore for points sampled from a ball (dim=3) in R^3
point_cloud_ball = np.array([random_array_1, random_array_2, random_array_3])
the_score = IsoScore.IsoScore(point_cloud_ball)
print(f"IsoScore for 100 points sampled from this ball in R^3 is {the_score}.")
```

### Isotropy in Contextualized Embeddings
We obtain contextualized word embeddings for the WikiText-2 corpus using: https://github.com/TideDancer/IsotropyContxt.

The embedding_results directory contains isotropy scores for BERT, DistilBERT, GPT and GPT-2. 

### Visuals 
Please consult ```visuals.ipynb ``` to quickly run tests and recreate figures.  


### Citing

If you would like to cite this work, please refer to:
```bibtex
@inproceedings{rudman-etal-2022-isoscore,
    title = "{I}so{S}core: Measuring the Uniformity of Embedding Space Utilization",
    author = "Rudman, William  and
      Gillman, Nate  and
      Rayne, Taylor  and
      Eickhoff, Carsten",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.262",
    doi = "10.18653/v1/2022.findings-acl.262",
    pages = "3325--3339",
}
```


### License

This project is licensed under the MIT License.
