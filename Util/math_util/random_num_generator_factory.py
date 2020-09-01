import Util.math_util.random_num_generator as generator
class AbstractNumberFactory():
    @staticmethod
    def get_num_generator():
        raise NotImplementedError

class NumberFactory(AbstractNumberFactory):
    @staticmethod
    def get_num_generator(distribution, **kwargs):
        if distribution is 'normal':
            return generator.NormalRandomNumGenerator(**kwargs)
        if distribution is 'binomial':
            return generator.BinomialRandomNumGenerator(**kwargs)
        if distribution is 'possion':
            return generator.PossionRandomNumGenerator(**kwargs)