import numpy as np
from sklearn.datasets import make_classification

samples=make_classification(n_samples=200,n_features=50000,n_informative=400)


X=samples[0]
Y=samples[1]
np.savetxt('train.txt',X)
np.savetxt('labels.txt',Y,fmt="%d")