import Util.math_util.random_num_generater as generater
class AbstractNumberFactory():
    @staticmethod
    def get_num_generater():
        raise NotImplementedError

class NumberFactory(AbstractNumberFactory):
    @staticmethod
    def get_num_generater(distribution, **kwargs):
        if distribution is 'normal':
            return generater.NormalRandomNumGenerater(**kwargs)