import numpy as np

class AbstractRandomNumGenerater():
    def get_ranodm_nun(self):
        raise NotImplementedError

class NormalRandomNumGenerater():
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

    

        
