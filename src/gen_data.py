import numpy as np
from sklearn.datasets import make_classification
import random

num_samples=20000
num_features=500
samples=make_classification(n_samples=num_samples,n_features=num_samples,n_informative=400)
X=samples[0]
Y=samples[1]

#Randomly shuffling samples for distributed SGD
'''
random_order=[i for i in range(20000)]
random.shuffle(random_order)
finalX=[]
finalY=[]
for i in random_order:
	print(i)
	finalX.append(X[i])
	finalY.append(Y[i])

'''

np.savetxt('train.txt',X)
np.savetxt('labels.txt',Y,fmt="%d")
