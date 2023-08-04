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

class PowerlawCutoff(Model):
    '''
        A 1D powerlaw with a cutoff
    '''
    log = True

    def __init__(self, theta, reffreq=150e6):
        self.theta = theta
        self.reffreq = reffreq
        self.model = np.vectorize(self._model)

    def _model(self, x):
        return self.theta[0]*10**(self.theta[1]*np.log10(x/self.reffreq))*np.exp(-x/self.theta[2])

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



class BrokenPowerlaw(Model):
    '''
        A broken powerlaw model, containing a set parameters theta that can be evaluated at a given x
    '''

    log = True
    def __init__(self, theta, reffreq=150e6):
        self.theta = theta
        self.reffreq = reffreq
        self.model = np.vectorize(self._model)

    def _model(self, x):
        I,s1,s2,vbreak = self.theta
        I_break = I*10**(s1*np.log10(vbreak/self.reffreq))
        if x > vbreak:
            return I_break*10**(s2*np.log10(x/vbreak))
        else:
            return I*10**(s1*np.log10(x/self.reffreq))
