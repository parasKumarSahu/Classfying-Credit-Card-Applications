import numpy as np
import random as rn
import matplotlib.pyplot as plt

def sigmoid(m):
#	m = 1/(1+np.exp(-1*m))
	return m

def standardize(X, mean, std):
	if mean == None and std == None:
		mean = list(np.mean(X, axis=0))
		std = list(np.std(X, axis=0))
	for i in range(len(X)): 
		for j in range(1, len(X[0])):
			X[i][j] = float(X[i][j] - mean[j])/float(std[j])
	return mean, std		


def newton_rapson(X, y, lamda, alpha, iterations):
	m = len(X)
	w = [0.01 for x in range(len(X[0]))]
	for i in range(iterations):
		z = np.matmul(X, np.transpose(w))
		Xt = np.transpose(X)
		h = sigmoid(z)
		grad = (1/m)*np.matmul(Xt, h-y) 
		H = (1/m)*np.matmul(Xt, np.matmul(np.matmul(np.diag(h),
		 np.diag(1-h)), X))+(2)*np.identity(len(w))*lamda
		tmp = np.linalg.solve(H, grad)
		w = [w[j] - tmp[j] for j in range(len(w))]
	return w

def gradient_descent(X, y, lamda, alpha, iterations):
	m = len(X)
	n = len(X[0])
	wt = np.transpose([0.01 for x in range(len(X[0]))])
	Xt = np.transpose(X)
	for i in range(iterations):
		diff =  np.matmul(X, wt) - np.transpose(y)
		mse = np.dot(diff.transpose(), diff)/m
#		print("iteration = ", i, " MSE: ", mse)
		wt = wt - alpha * ((np.matmul(Xt, diff) / m) + (lamda/m)*wt)
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
#	ans = sigmoid(ans)
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

def apply_polynomial(w, degree, x, y):
	z = None
	if degree <= 1:
		z = (w[0]+x+y)
	elif degree == 2:
		z = (w[0]+w[1]*x+w[2]*y+w[3]*(x**2)+w[4]*x*y+w[5]*(y**2))
	elif degree == 3:
		z = (w[0]+w[1]*x+w[2]*y+w[3]*(x**2)+w[4]*x*y+w[5]*(y**2)+w[6]*(x**3)+w[7]*(x**2)*y+w[8]*x*(y**2)+w[9]*(y**3))
	else:
		z = (w[0]+w[1]*x+w[2]*y+w[3]*(x**2)+w[4]*x*y+w[5]*(y**2)+w[6]*(x**3)+w[7]*(x**2)*y+w[8]*x*(y**2)+w[9]*(y**3)
			+w[10]*(x**4)+w[11]*(x**3)*y+w[12]*(x**2)*(y**2)+w[13]*x*(y**3)+w[14]*(y**4))

	return z	

#Main Function

f = open("credit.txt", "r")
#output
y = []
#Training Data
X = []
#lambda
lamda = [i for i in range(50)]
#Learning rate
alpha = 0.001
#Fraction of training/validation set
frac = 0.4
#Degree
degree = 2
#iterations
iterations = 10000

for i in f:
	tmp = i[:-1].split(",")
	y.append(int(tmp[-1]))
	X.append([1]+[float(i) for i in tmp[:-1]])

X_backup = X
y_backup = y

for d in range(1, 5):
	X = X_backup
	y = y_backup
	X = get_polynomial(X, d)
	X_train, X_test, y_train, y_test = split_train_test(X, y, frac)
	mean, std = standardize(X_train, None, None)
	standardize(X_test, mean, std)
	errors = []
	errors_w = []
	errors_grad = []
	errors_w_grad = []
	for l in lamda:
		w = newton_rapson(X_train, y_train, l, alpha, 100)
		ans =  mylogridgeregeval(X_test, w)
		errors.append(meansquarederr(ans, y_test))
		errors_w.append(w)

	w = errors_w[errors.index(min(errors))]
	plt.scatter([i[1] for i in X], [i[2] for i in X], marker='o', c=y, s=25, edgecolor='k')	
	xlist = np.linspace(-3, 3, 1000)
	ylist = np.linspace(-3, 3, 1000)
	x,y = np.meshgrid(xlist, ylist)
	z = apply_polynomial(w, d, x, y)
	plt.contour(x, y, z, [0])
	plt.xlabel('Feature 1') 
	plt.ylabel('Feature 2') 
	plt.title('Found using Newton Rapson  Lambda: '+str(errors.index(min(errors)))+" Error: "+str(min(errors))) 
	plt.savefig("newton/degree"+str(d)+".png")
	plt.clf()

for d in range(1, 5):
	X = X_backup
	y = y_backup
	X = get_polynomial(X, d)
	X_train, X_test, y_train, y_test = split_train_test(X, y, frac)
	mean, std = standardize(X_train, None, None)
	standardize(X_test, mean, std)
	errors_grad = []
	errors_w_grad = []
	for l in lamda:
		w = gradient_descent(X_train, y_train, l, alpha, iterations)
		ans =  mylogridgeregeval(X_test, w)
		errors_grad.append(meansquarederr(ans, y_test))
		errors_w_grad.append(w)

	w = errors_w_grad[errors_grad.index(min(errors_grad))]
	plt.scatter([i[1] for i in X], [i[2] for i in X], marker='o', c=y, s=25, edgecolor='k')	
	xlist = np.linspace(-3, 3, 1000)
	ylist = np.linspace(-3, 3, 1000)
	x,y = np.meshgrid(xlist, ylist)
	z = apply_polynomial(w, d, x, y)
	plt.contour(x, y, z, [0])
	plt.xlabel('Feature 1') 
	plt.ylabel('Feature 2') 
	plt.title('Found using Gradient Descent  Lambda: '+str(errors_grad.index(min(errors_grad)))+" Error: "+str(min(errors_grad))) 
	plt.savefig("grad/degree"+str(d)+".png")
	plt.clf()	

X = X_backup
y = y_backup
X = get_polynomial(X, 2)
X_train, X_test, y_train, y_test = split_train_test(X, y, frac)
mean, std = standardize(X_train, None, None)
standardize(X_test, mean, std)
w = gradient_descent(X_train, y_train, 4, alpha, iterations)
ans =  mylogridgeregeval(X_test, w)
plt.scatter([i[1] for i in X], [i[2] for i in X], marker='o', c=y, s=25, edgecolor='k')	
xlist = np.linspace(-3, 3, 1000)
ylist = np.linspace(-3, 3, 1000)
x,y = np.meshgrid(xlist, ylist)
z = apply_polynomial(w, 2, x, y)
plt.contour(x, y, z, [0])
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 
plt.title('Underfit Condition  Lambda: 4 Degree: 2') 
plt.savefig("underfit.png")
plt.clf()		

frac = 0.1

X = X_backup
y = y_backup
X = get_polynomial(X, 4)
X_train, X_test, y_train, y_test = split_train_test(X, y, frac)
mean, std = standardize(X_train, None, None)
standardize(X_test, mean, std)
w = gradient_descent(X_train, y_train, 0, alpha, iterations)
ans =  mylogridgeregeval(X_test, w)
plt.scatter([i[1] for i in X], [i[2] for i in X], marker='o', c=y, s=25, edgecolor='k')	
xlist = np.linspace(-3, 3, 1000)
ylist = np.linspace(-3, 3, 1000)
x,y = np.meshgrid(xlist, ylist)
z = apply_polynomial(w, 4, x, y)
plt.contour(x, y, z, [0])
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 
plt.title('Overfit Condition  Lambda: 0 Degree: 4') 
plt.savefig("overfit.png")
plt.clf()		
