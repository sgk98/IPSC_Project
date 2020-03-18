import numpy as np
import math
import random
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def solve(X,Y,lm,n_iter=100):
	C=len(Y)
	W=np.array([0 for i in range(len(X[0]))])
	print("len ", len(X[0]))
	fobj = open("choice", "r")
	choices = fobj.readlines()
	# print(choices)
	for it in range(n_iter):
		#print(it)
		eta=1.0/(lm*(it+1))
		choice=int(choices[6*it])
		print (choice)
		x,y=X[choice],Y[choice]
		out = 0
		# for i in range(len(x)):
			# out += W[i] * x[i]
		out=np.dot(W.T,x)
		print("check1 ", out)
		print("check2 ", y)
		print("check3 ", y*out)
		if y*out >= 1:
			print("here1")
			W = (1-eta*lm)*W
		else:
			print("here2")
			W = (1-eta*lm)*W + (eta*y)*x
		print("check4 ", W[0], W[1])



	return W


if __name__=="__main__":

	# separable = False
	# while not separable:
	    # samples = make_classification(n_samples=20000,n_features=500,n_informative=400)
	    # red = samples[0][samples[1] == 0]
	    # blue = samples[0][samples[1] == 1]
	    # separable = True
	
	X = np.loadtxt("train.txt")
	Y = np.loadtxt("labels.txt")
	# X=samples[0]
	# Y=samples[1]

	# np.savetxt('train.txt',X)
	# np.savetxt('labels.txt',Y,fmt="%d")
	copyX=[]
	for i in range(len(Y)):
		tmp=X[i]
		tmp=np.append(tmp,1)
		copyX.append(tmp)
		if Y[i]==0:
			Y[i]=-1
	copyX=np.array(copyX)
	X=copyX
	W=solve(X,Y,lm=1,n_iter=5000)
	#print(W)
	correct=0
	total=0

	for i in range(len(X)):
		res=np.dot(W.T,X[i])
		if res*Y[i]>=0:
			correct+=1.0
		total+=1.0
	print(correct/total)


