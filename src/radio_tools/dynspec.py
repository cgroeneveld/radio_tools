import dynesty
import multiprocessing as mp
import numpy as np
from . import lib_dyn


class Ensemble():
    '''
        Ensemble of scenarios, to be used for comparison
    '''

    def __init__(self, scenarios, x=None, y=None, yerr=None, nthreads=mp.cpu_count()):
        self.scenarios = scenarios
        self.nthreads = nthreads
        if x is not None:
            self.add_data(x, y, yerr)

    def add_scenario(self, scenario):
        self.scenarios.append(scenario)

    def add_data(self, x, y, yerr):
        for scenario in self.scenarios:
            assert len(x) == len(y) == len(yerr)
            scenario.set_data(x, y, yerr)

    def run(self):
        evidences = []
        results = []
        for scenario in self.scenarios:
            scenario.nthreads = self.nthreads
            res = scenario.run()
            evidences.append(res.logz[-1])
            results.append(res)
        print('--------------------')
        print('Best evidence: ', np.max(evidences))
        print('Best scenario: ', np.argmax(evidences))
        print('Model parameters: ', results[np.argmax(evidences)].samples[-1])
        self.results = results
        self.evidences = evidences
        self.model = self.scenarios[np.argmax(evidences)].model(results[np.argmax(evidences)].samples[-1])
        return results[np.argmax(evidences)]

    @property
    def cov(self):
        best_scenario = self.scenarios[np.argmax(self.evidences)]
        return best_scenario.cov

class MultiPowerlaw(Ensemble):
    '''
        Try out power laws of decreasing order (starting from 
        the order specified by the bounds provided)
    '''

    def __init__(self, boundlist, x=None, y=None, yerr=None):
        scenarios = []
        for i in range(len(boundlist)):
            scenarios.append(PowerlawScenario(boundlist[0:i+1]))
        super().__init__(scenarios, x, y, yerr)


class Scenario():
    '''
        Contains a model, the required prior and the loglikelihood function.
        This is enough for dynesty to run - but not useful on itself.
        Parent class to all other scenarios
    '''
    prior_type = 'linear'
    model = lib_dyn.Model  # Placeholder model-
    nwalkers = 300
    bound = 'single'
    sample = 'auto'
    name = 'Model'

    def __init__(self, bounds, x=None, y=None, yerr=None, nthreads=1):
        self.boundlist = []
        for bound in bounds:
            self.boundlist.append([bound[0], bound[1], self.prior_type])
        self.prior = lib_dyn.Prior(self.boundlist)
        self.x = x
        self.y = y
        self.order = len(bounds)
        self.nthreads = nthreads
        self.yerr = yerr
        self.ndim = len(self.boundlist)

    def set_data(self, x, y, yerr):
        self.x = x
        self.y = y
        if yerr is not ModuleNotFoundError:
            self.yerr = yerr
        else:
            self.yerr = 0.2*self.y

    def loglikelihood(self, theta):
        model = self.model(theta)
        return -0.5*np.sum((self.y-model(self.x))**2/self.yerr**2)

    def run(self):
        sampler = dynesty.DynamicNestedSampler(
            self.loglikelihood, self.prior, bootstrap=0, pool=mp.Pool(self.nthreads), ndim=self.ndim, nlive=self.nwalkers, bound=self.bound, sample=self.sample, queue_size=self.nthreads)
        sampler.run_nested()
        self.results = sampler.results
        return sampler.results

    @property
    def cov(self):
        if self.results != None:
            mean, cov = dynesty.utils.mean_and_cov(self.results.samples, self.results.importance_weights())
            return cov
        else:
            raise ValueError('No results found. Run the scenario first.')

class PowerlawScenario(Scenario):
    '''
        Scenario to fit a powerlaw to your data
        Can be tuned to any arbitrary order
    '''
    prior_type = 'linear'
    model = lib_dyn.Powerlaw

    def __init__(self, bounds, x=None, y=None, yerr=None):
        self.order = len(bounds)
        super().__init__(bounds, x, y, yerr)
        self.boundlist[0][2] = 'log'
        self.prior = lib_dyn.Prior(self.boundlist)
        self.name = 'Powerlaw order ' + str(self.order)


class CutoffScenario(Scenario):
    '''
        Scenario to fit a powerlaw with a cutoff to your data
    '''
    prior_type = 'linear'
    model = lib_dyn.PowerlawCutoff
    name = 'Powerlaw with cutoff'
    bound = 'single'
    sample = 'rwalk'

    def __init__(self, bounds, x=None, y=None, yerr=None):
        assert len(bounds) == 3
        super().__init__(bounds, x, y, yerr)
        self.boundlist[0][2] = 'log'
        self.prior = lib_dyn.Prior(self.boundlist)
        self.boundlist[2][2] = 'log'


class BrokenPowerlaw(Scenario):
    '''
        Scenario to fit a broken powerlaw to your data
    '''
    prior_type = 'linear'
    model = lib_dyn.BrokenPowerlaw
    bound = 'single'
    sample = 'rwalk'

    def __init__(self, bounds, x=None, y=None, yerr=None):
        assert len(bounds) == 4
        super().__init__(bounds, x, y, yerr)
        self.boundlist[0][2] = 'log'
        self.boundlist[3][2] = 'log'
        self.prior = lib_dyn.Prior(self.boundlist)
        self.name = 'Broken powerlaw'

    def run(self):
        sampler = dynesty.DynamicNestedSampler(
            self.loglikelihood, self.prior, ndim=self.ndim, nlive=self.nwalkers, pool=mp.Pool(96), queue_size=96, bound=self.bound, sample=self.sample)
        sampler.run_nested()
        return sampler.results
