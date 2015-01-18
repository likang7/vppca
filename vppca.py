# coding:utf-8

import numpy as np
from scipy.special import digamma, gammaln
from pylab import *

class VPPCA:
    def __init__(self, K=None, n_iter=1000, verbose=False, thresh = 1e-2):
        self.K = K
        self.n_iter = n_iter
        self.thresh = thresh
        self.verbose = verbose

    def _init_paras(self, N, P, K):
        self.mean_A = np.random.normal(loc=0.0, scale=1.0, size=(P, K))
        self.prec_A = np.eye(K)

        self.a_alpha = np.ones(K)
        self.b_alpha = np.ones(K)
        self.alpha = self.a_alpha / self.b_alpha

        self.a_psi = 1
        self.b_psi = 1
        self.psi = self.a_psi / self.b_psi

        self.mean_s = np.random.normal(loc=0.0, scale=1.0, size=(N, K))
        self.prec_s = np.eye(K)

        self.mean_mu = np.zeros(P)
        self.prec_mu = np.eye(P)

        self.a_alpha0 = 1e-3
        self.b_alpha0 = 1e-3
        self.a_psi0 = 1e-3
        self.b_psi0 = 1e-3
        self.mu_prec0 = 1e-3

    def _update_s(self, X, K, P):
        prec_s = np.linalg.inv(np.eye(K) + self.psi * (self.mean_A.T.dot(self.mean_A) + P*self.prec_A))
        precS_A_psi = prec_s.dot(self.mean_A.T) * self.psi
        mean_s = (X - self.mean_mu).dot(precS_A_psi.T)
        return mean_s, prec_s

    def _update_mu(self, X, K):
        N, P = X.shape
        cov = self.mu_prec0 + N * self.psi
        self.prec_mu = (cov ** -1) * np.eye(P)

        mean_mu = np.sum(X - self.mean_s.dot(self.mean_A.T), axis=0)
        self.mean_mu = self.prec_mu.dot(mean_mu) * self.psi

    def _update_A(self, X, P):
        N = X.shape[0]
        Eq_ss = self.mean_s.T.dot(self.mean_s) + N*self.prec_s
        self.prec_A = np.linalg.inv(np.diag(self.alpha) + self.psi * Eq_ss)
        for j in xrange(P):
            self.mean_A[j] = self.prec_A.dot(self.mean_s.T.dot(X[:, j] - self.mean_mu[j])) * self.psi

    def _update_alpha(self, P, K):
        self.a_alpha[:] = self.a_alpha0 + 0.5*P
        self.b_alpha = self.b_alpha0 + 0.5*(np.sum(self.mean_A**2, 0) + P*np.diag(self.prec_A))
        self.alpha = self.a_alpha / self.b_alpha

    def _update_psi(self, X, N, P):
        self.a_psi = self.a_psi0 + 0.5*N*P
        tsum = np.sum(X**2) + N*(self.mean_mu.T.dot(self.mean_mu) + np.trace(self.prec_mu))
        Eq_aa = self.mean_A.T.dot(self.mean_A) + P*self.prec_A
        for i in xrange(N):
            Eq_ss = np.outer(self.mean_s[i], self.mean_s[i]) + self.prec_s
            tsum += np.trace(Eq_aa.dot(Eq_ss)) 
            tsum += 2 * self.mean_mu.T.dot(self.mean_A).dot(self.mean_s[i])
            tsum -= 2*X[i].T.dot(self.mean_A).dot(self.mean_s[i])
            tsum -= 2*self.X[i].T.dot(self.mean_mu)
        self.b_psi = self.b_psi0 + 0.5*tsum
        self.psi = self.a_psi / self.b_psi

    def lower_bound(self, X, N, P, K):
        lb = 0
        #S
        lps = 0
        for n in xrange(N):
            lps += self.mean_s[n].T.dot(self.mean_s[n]) + np.trace(self.prec_s)
        lps = -.5 * lps
        sign, logdet = np.linalg.slogdet(self.prec_s)
        assert sign > 0
        lqs = -0.5 * N * (K + sign * logdet)
        lb += lps - lqs

        #mu
        lpmu = .5 * P * np.log(self.mu_prec0) 
        lpmu -= .5 * self.mu_prec0 * (self.mean_mu.T.dot(self.mean_mu) + np.trace(self.prec_mu))
        sign, logdet = np.linalg.slogdet(self.prec_mu)
        assert sign > 0
        lqmu = -.5 * (P + sign * logdet)
        lb += lpmu - lqmu

        #A
        lpA = .5 * P * np.sum(digamma(self.a_alpha) - np.log(self.b_alpha))
        lpA -= .5 * np.sum(np.multiply(self.alpha, np.sum(self.mean_A**2, axis=0) + np.diag(P * self.prec_A)))
        sign, logdet = np.linalg.slogdet(self.prec_A)
        assert sign > 0
        lqA = -.5 * P * (K + sign*logdet)
        lb += lpA - lqA

        #alpha
        lp_alpha = np.sum(-gammaln(self.a_alpha0) + self.a_alpha0*np.log(self.b_alpha0) + \
            (self.a_alpha0 - 1)*(digamma(self.a_alpha) - np.log(self.b_alpha)) - \
            self.b_alpha0*self.alpha)
        lq_alpha = np.sum(-gammaln(self.a_alpha) + (self.a_alpha - 1)*digamma(self.a_alpha) - \
            self.a_alpha + np.log(self.b_alpha))
        lb += lp_alpha - lq_alpha

        #psi
        lp_psi = -gammaln(self.a_psi0) + self.a_psi0*np.log(self.b_psi0) + \
            (self.a_psi0 - 1)*(digamma(self.a_psi) - np.log(self.b_psi)) - \
            self.b_psi0 * self.psi
        lq_psi = -gammaln(self.a_psi) + (self.a_psi - 1)*digamma(self.a_psi) - \
            self.a_psi + np.log(self.b_psi)
        lb += lp_psi - lq_psi

        #x
        lpx = -.5 * N * P * (np.log(2*np.pi) - (digamma(self.a_psi) - np.log(self.b_psi)))
        lpx -= self.psi*(self.b_psi - self.b_psi0)
        # Eq_aa = self.mean_A.T.dot(self.mean_A) + self.prec_A
        # Eq_mumu = self.mean_mu.T.dot(self.mean_mu) + np.trace(self.prec_mu)
        # for i in xrange(N):
        #   tsum = np.sum(X[i]**2) + Eq_mumu
        #   Eq_ss = np.outer(self.mean_s[i], self.mean_s[i]) + self.prec_s
        #   tsum += np.trace(Eq_aa.dot(Eq_ss)) 
        #   tsum += 2 * self.mean_mu.T.dot(self.mean_A).dot(self.mean_s[i])
        #   tsum -= 2*X[i].T.dot(self.mean_A).dot(self.mean_s[i])
        #   tsum -= 2*self.X[i].T.dot(self.mean_mu)
        #   lpx -= .5 * self.psi * tsum

        lb += lpx
        return lb

    def fit(self, X):
        self.lbs = []
        N, P = X.shape
        K = self.K
        if K is None:
            K = P

        self.X = np.asarray(X)
        self._init_paras(N, P, K)
        self.converge = False
        for i in xrange(self.n_iter):
            self._update(X, N, P, K)
            if i >= 1 and (self.lbs[-1] - self.lbs[-2] < self.thresh):
                self.converge = True
                break
        if self.verbose:
            ax = np.arange(len(self.lbs))
            ay = np.asarray(self.lbs)
            plot(ax,ay)
            show()

    def _update(self, X, N, P, K):
        #E STEP
        self.mean_s, self.prec_s = self._update_s(X, K, P)
        #M STEP
        self._update_mu(X, K)
        self._update_A(X, P)
        self._update_alpha(P, K)
        self._update_psi(X, N, P)
        self.lbs.append(self.lower_bound(X, N, P, K))

        if self.verbose:
            print 'mse: ', self.mse()
            if len(self.lbs) >= 2:
                print 'eps: ', self.lbs[-1] - self.lbs[-2]

    def update(self):
        X = self.X
        N, P = X.shape
        K = self.mean_s.shape[1]
        self._update(X, N, P, K)

    def transform(self, X):
        X = np.asarray(X)
        K = self.mean_s.shape[1]
        P = X.shape[1]
        X_transformed, _ = self._update_s(X, K, P)
        return X_transformed

    def recover(self, s = None):
        if s is None:
            s = self.mean_s
        return s.dot(self.mean_A.T) + self.mean_mu

    def mse(self):
        d = self.X - self.recover()
        d = d.ravel()
        N = self.X.shape[0]
        return (d.dot(d)/N)
