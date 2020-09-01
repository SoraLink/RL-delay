import numpy as np

class AbstractRandomNumGenerator():
    def get_ranodm_nun(self):
        raise NotImplementedError

class NormalRandomNumGenerator(AbstractRandomNumGenerator):
    def __init__(self, **kwargs):
        self.max = kwargs['max']
        self.min = kwargs['min']
        self.mu = kwargs['mu']
        self.sigma = kwargs['sigma']

    def get_random_num(self):
        res = -1
        while(res<self.min or res>self.max):
            res = np.random.normal(self.mu, self.sigma)
            res = int(res)
        return res

class BinomialRandomNumGenerator(AbstractRandomNumGenerator):
    def __init__(self, **kwargs):
        self.n = kwargs['n']
        self.p = kwargs['p']
    
    def get_random_num(self):
        return np.random.binomial(self.n, self.p)
    
class PossionRandomNumGenerator(AbstractRandomNumGenerator):
    def __init__(self, **kwargs):
        self.lam = kwargs['lam']

    def get_random_num(self):
        return np.random.poisson(self.lam)
        
        

    

        
