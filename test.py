from vppca import VPPCA
from utils.hinton import hinton
import matplotlib.pylab as plt
import numpy as np

def simulate():
	N = 50
	P = 10
	K = 4

	s = np.random.normal(0, 1, size=(N, K))
	alpha = np.random.gamma(1, 1, size=(K))
	A = np.empty((P, K))
	for k in xrange(K):
		A[:, k] = np.random.multivariate_normal(np.ones(P), (alpha[k]**-1) * np.eye(P))
	y = np.empty((N, P))
	mu = np.random.normal(0, 1, size=(P))
	for i in xrange(N):
		y[i] = np.random.multivariate_normal(A.dot(s[i])+mu, np.eye(P))
	return y

def _main():
	np.random.seed(0)
	X = simulate()
	K = 9

	model = VPPCA(K = K, n_iter = 1000, verbose = False, thresh=1e-3)
	model.fit(X)
	hinton(model.mean_A)
	plt.show()

_main()