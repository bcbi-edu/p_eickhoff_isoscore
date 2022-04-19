# IsoScore: Measuring the Uniformity of Embedding Space Utilization
 *by William Rudman, Nate Gillman, Taylor Rayne, and Carsten Eickhoff*

IsoScore is a tool that measures how uniformly a point cloud utilizes the Euclidian space that it sits inside of. 
See the (original paper (https://arxiv.org/abs/2108.07344)) appearing in the Findings of ACL 2022 for more information. 
This repository contains the Python3 implementation of IsoScore.

### How to install

The only dependencies are `numpy` and `sklearn`.

```
pip install IsoScore
```

### How to use

If you want to compute the IsoScore for a point cloud <img src="https://render.githubusercontent.com/render/math?math=X">  that sits inside <img src="https://render.githubusercontent.com/render/math?math=\mathbb R^n">, then <img src="https://render.githubusercontent.com/render/math?math=X"> must be a `numpy` array of shape <img src="https://render.githubusercontent.com/render/math?math=(n,m)">, where <img src="https://render.githubusercontent.com/render/math?math=X"> contains <img src="https://render.githubusercontent.com/render/math?math=m"> points.
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
@Article{Rudman-Gillman-Rayne-Eickhoff-IsoScore,
    title = "IsoScore: Measuring the Uniformity of Vector Space Utilization",
    author =    {William Rudman and
                Nate Gillman and 
                Taylor Rayne and 
                Carsten Eickhoff},
    month = aug,
    year = "2021",
    url = "https://arxiv.org/abs/2108.07344",
}
```


### License

This project is licensed under the MIT License.
