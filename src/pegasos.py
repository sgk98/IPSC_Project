import numpy as np
import math
import random
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def solve(X,Y,lm,n_iter=100):
	C=len(Y)
	W=np.array([0 for i in range(len(X[0]))])

	for it in range(n_iter):
		#print(it)
		eta=1.0/(lm*(it+1))
		choice=random.randint(0,C-1)
		x,y=X[choice],Y[choice]
		out=np.dot(W.T,x)
		if y*out >= 1:
			W = (1-eta*lm)*W
		else:
			W = (1-eta*lm)*W + (eta*y)*x



	return W


if __name__=="__main__":

	separable = False
	while not separable:
	    samples = make_classification(n_samples=20000,n_features=500,n_informative=400)
	    red = samples[0][samples[1] == 0]
	    blue = samples[0][samples[1] == 1]
	    separable = True
	
	X=samples[0]
	Y=samples[1]
	copyX=[]
	for i in range(len(Y)):
		tmp=X[i]
		tmp=np.append(tmp,1)
		copyX.append(tmp)
		if Y[i]==0:
			Y[i]==-1
	copyX=np.array(copyX)
	X=copyX
	W=solve(X,Y,lm=1,n_iter=20000)
	#print(W)
	correct=0
	total=0

	for i in range(len(X)):
		res=np.dot(W.T,X[i])
		if res*Y[i]>=0:
			correct+=1.0
		total+=1.0
	print(correct/total)


