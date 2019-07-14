#import DynamicsPriorGMM
import numpy as np

from dynamics.dynamics import Dynamics
from algorithm.algorithm_utils import gauss_fit_joint_prior

class DynamicsLRPrior(Dynamics):
    """ Dynamics with linear regression, with arbitrary prior. """
    def __init__(self, hyperparams):
        #import pdb; pdb.set_trace()
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self.prior = self._hyperparams['prior']['type'](self._hyperparams['prior'])
        #DynamicsPriorGMM.DynamicsPriorGMM()

    def update_prior(self, samples):
        """ Update dynamics prior. """
        #TODO
        #import pdb; pdb.set_trace()
        X = samples.get_X()
        U = samples.get_U()
        self.prior.update(X, U)

    def get_prior(self):
        """ Return the dynamics prior. """
        return self.prior

    #def gauss_fit_joint_prior(self, pts, mu0, Phi, m, n0, dwts, dX, dU, sig_reg):
    #    """ Perform Gaussian fit to data with a prior. """
    #    # Build weights matrix.
    #    D = np.diag(dwts)
    #    # Compute empirical mean and covariance.
    #    mun = np.sum((pts.T * dwts).T, axis=0)
    #    diff = pts - mun
    #    empsig = diff.T.dot(D).dot(diff)
    #    empsig = 0.5 * (empsig + empsig.T)
    #    # MAP estimate of joint distribution.
    #    N = dwts.shape[0]
    #    #import pdb; pdb.set_trace()
    #    mu = mun
    #    sigma = (N * empsig + Phi + (N * m) / (N + m) *
    #    #sigma = (N * empsig + Phi + (n0 * m) / (n0 + m) *
    #            np.outer(mun - mu0, mun - mu0)) / (N + n0)
    #    sigma = 0.5 * (sigma + sigma.T)
    #    #import pdb; pdb.set_trace()
    #    # Add sigma regularization.
    #    sigma += sig_reg
    #    # Conditioning to get dynamics.
    #    fd = np.linalg.solve(sigma[:dX, :dX], sigma[:dX, dX:dX+dU]).T
    #    fc = mu[dX:dX+dU] - fd.dot(mu[:dX])
    #    dynsig = sigma[dX:dX+dU, dX:dX+dU] - fd.dot(sigma[:dX, :dX]).dot(fd.T)
    #    dynsig = 0.5 * (dynsig + dynsig.T)
    #    return fd, fc, dynsig

    #TODO: Merge this with DynamicsLR.fit - lots of duplicated code.
    def fit(self, X, U):
        """ Fit dynamics. """
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.prior.eval(dX, dU, Ys)
            #import pdb; pdb.set_trace()
            sig_reg = np.zeros((dX+dU+dX, dX+dU+dX))
            sig_reg[it, it] = 1e-6
            Fm, fv, dyn_covar = gauss_fit_joint_prior(Ys,
                        mu0, Phi, mm, n0, dwts, dX+dU, dX, sig_reg)
            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar
        return self.Fm, self.fv, self.dyn_covar
