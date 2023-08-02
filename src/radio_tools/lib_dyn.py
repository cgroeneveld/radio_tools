import dynesty
import numpy as np
import multiprocessing as mp

class _PriorElement():
    def __init__(self, min, max, scaletype):
        self.min = min
        self.max = max
        self.scaletype = scaletype

    def __call__(self, u):
        if self.scaletype == 'log':
            return np.exp(u*(np.log(self.max)-np.log(self.min))+np.log(self.min))
        elif self.scaletype == 'linear':
            return u*(self.max-self.min)+self.min
        else:
            raise ValueError('scaletype must be either "log" or "linear"')


class Prior():
    def __init__(self, priorlist):
        self.priors = []
        for el in priorlist:
            self.priors.append(_PriorElement(*el))

    def __call__(self, u):
        return np.array([priorel(u_el) for priorel,u_el in zip(self.priors,u)])


class Model():
    '''
        A model, containing a set parameters theta that can be evaluated at a given x
    '''
    def model(self,x):
        return 0 # Placeholder function

    def __init__(self, theta):
        self.theta = theta

    def __call__(self, x):
        return self.model(x)

    theta = []
    ndim = len(theta)


class LinearModel(Model):
    '''
        A linear model, containing a set parameters theta that can be evaluated at a given x
    '''
    log = False

    def model(self, x):
        return self.theta[0]*x+self.theta[1]


class QuadraticModel(Model):
    '''
        A quadratic model, containing a set parameters theta that can be evaluated at a given x
    '''
    log = False

    def model(self, x):
        return self.theta[0]*x**2+self.theta[1]*x+self.theta[2]


class CubicModel(Model):
    '''
        A cubic model, containing a set parameters theta that can be evaluated at a given x
    '''
    log = False

    def model(self, x):
        return self.theta[0]*x**3+self.theta[1]*x**2+self.theta[2]*x+self.theta[3]


class Powerlaw(Model):
    '''
        A powerlaw model, containing a set parameters theta that can be evaluated at a given x
    '''

    def __init__(self, theta, reffreq=150e6):
        self.theta = theta
        self.reffreq = reffreq
        self.order = len(theta)-1
        self.powerlaw_terms = np.array(theta[1:])
        self.order_list = np.arange(self.order)+1
        self.model = np.vectorize(self._model)

    log = True

    def _model(self, x):
        return self.theta[0]*10**(np.sum(np.log10(x/self.reffreq)**self.order_list*self.powerlaw_terms))


def log_likelihood(model, x, y, yerr):
    '''
        Calculate the log likelihood of a model given data and errors
        Assumes Gaussian errors
    '''
    return -0.5*np.sum((y-model(x))**2/yerr**2)


def evaluateSampler(model, x, y, yerr, priorlist, loglikelihood, nwalkers=100, ndim=None, nthreads=1, **kwargs):
    '''
        Evaluate a sampler for a given model, data, and priors
        Returns the sampler
    '''
    if ndim is None:
        ndim = model.ndim
    sampler = dynesty.NestedSampler(loglikelihood, priorlist, ndim, nlive=nwalkers, bound='multi', sample='unif', pool=dynesty.utils.get_pool(nthreads), queue_size=nthreads, **kwargs)
    sampler.run_nested()
    return sampler
