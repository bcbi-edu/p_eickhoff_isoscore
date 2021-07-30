# IsoScore

This contains the Python3 implementation of IsoScore, which was originally
introduced in the 2021 paper by William Rudman, Nate Gillman, Taylor Rayne, and 
Carsten Eickhoff. IsoScore is a tool which measures how uniformly a point cloud 
utilizes the Euclidian space that it sits inside of. See the original paper for more information.


### How to use

The only dependencies are `numpy` and `sklearn`.

```python3
import numpy as np
from IsoScore import IsoScore

# Computing the IsoScore for a fuzzy ball in R^3
point_cloud_isotropic = np.random.normal(size=(3,100))
the_score = IsoScore.IsoScore(point_cloud_isotropic)
print(f"The IsoScore for 100 points sampled from this Gaussian ball in R^3 is {the_score},")

# Computing the IsoScore for points sampled from the line t \mapsto (t, 2t, 3t) in R^3
random_array = np.random.normal(size=100)
point_cloud_anisotropic = np.array([random_array, 2*random_array, 3*random_array])
the_score = IsoScore.IsoScore(point_cloud_anisotropic)
print(f"and the IsoScore for 100 points sampled from this line in R^3 is {the_score}.")
```

### License

This project is licensed under the MIT License.

