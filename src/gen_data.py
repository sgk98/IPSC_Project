import numpy as np
from sklearn.datasets import make_classification

samples=make_classification(n_samples=20000,n_features=500,n_informative=400)


X=samples[0]
Y=samples[1]
np.savetxt('train.txt',X)
np.savetxt('labels.txt',Y,fmt="%d")