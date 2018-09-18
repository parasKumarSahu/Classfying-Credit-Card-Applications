import numpy as np
import random as rn
import matplotlib.pyplot as plt


def standardize(X, mean, std):
	if mean == None and std == None:
		mean = list(np.mean(X, axis=0))
		std = list(np.std(X, axis=0))
	for i in range(len(X)): 
		for j in range(1, len(X[0])):
			X[i][j] = float(X[i][j] - mean[j])/float(std[j])
	return mean, std		

def gradient_descent(X, y, lamda, alpha, iterations):
	m = len(X)
	n = len(X[0])
	wt = np.transpose([1 for x in range(len(X[0]))])
	Xt = np.transpose(X)
	for i in range(iterations):
		diff = np.dot(X, wt) - np.transpose(y)
		mse = np.dot(diff.transpose(), diff)/m
#		print("iteration = ", i, " MSE: ", mse)
		wt = wt - alpha * ((np.dot(Xt, diff) / m) + (lamda/m)*wt)
	return np.transpose(wt)	

def split_train_test(X, y, frac):
	X_train = []
	y_train = []
	X_test = []
	y_test = []
	for i in range(len(X)):
		if rn.random() < frac:
			X_train.append(X[i])
			y_train.append(y[i])
		else:
			X_test.append(X[i])
			y_test.append(y[i])	
	return X_train, X_test, y_train, y_test		

def mylogridgeregeval(X_test, w):
	ans = np.matmul(X_test, np.transpose(w))
	for i in range(len(ans)):
		if ans[i] < 0.5:
			ans[i] = 0
		else:
			ans[i] = 1	
	return ans		

def meansquarederr(ans, y_test):
	return ((ans - y_test) ** 2).mean()

def get_polynomial(X, degree):
	X_poly = []
	X1 = [x[1] for x in X]
	X2 = [x[2] for x in X]
	for i in range(0, degree+1): 
		for j in range(i+1):
			X_poly.append( list( np.float_power( X1, (i-j) ) * np.float_power(X2, j) ) )
	return np.transpose(X_poly)		

#Main Function

f = open("credit.txt", "r")
#output
y = []
#Training Data
X = []
#lambda
lamda = 1
#Learning rate
alpha = 0.05
#Fraction of training/validation set
frac = 0.2

for i in f:
	tmp = i[:-1].split(",")
	y.append(int(tmp[-1]))
	X.append([1]+[float(i) for i in tmp[:-1]])


X_train, X_test, y_train, y_test = split_train_test(X, y, frac)
mean, std = standardize(X_train, None, None)
standardize(X_test, mean, std)


w = gradient_descent(X_train, y_train, lamda, alpha, 100000)
ans =  mylogridgeregeval(X_test, w)
print(meansquarederr(ans, y_test))

#print(y)
'''
X = get_polynomial(X, 4)
X_train, X_test, y_train, y_test = split_train_test(X, y, frac)
mean, std = standardize(X_train, None, None)
standardize(X_test, mean, std)

w = gradient_descent(X_train, y_train, lamda, alpha, 100000)
ans =  mylogridgeregeval(X_test, w)
print(meansquarederr(ans, y_test))
'''
plt.scatter([i[1] for i in X], [i[2] for i in X], marker='o', c=y, s=25, edgecolor='k')	
plt.show()
