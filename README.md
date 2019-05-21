# fast_relaxed_dtw
This is an combination of fast dtw and relaxed dtw.

#### Usage

```
import numpy as np
from scipy.spatial.distance import euclidean
from fast_relaxed_dtw import fast_relaxed_dtw

x = np.array([1, 1, 2, 3, 2, 0])
y = np.array([2, 3, 3, 4, 5, 4])
distance, path = fast_relaxed_dtw(x, y, dist=euclidean, r=3)

``` 
