from random import Random
class Dataset():
    def __init__(self, max_size, ):
        self.max_size = max_size
        self.pairs = []

    def add_instance(self, pair):
        if len(self.pairs) >= self.max_size:
            self.pairs[self.index] = pair
            if self.index < self.max_size:
                self.index += 1
            else:
                self.index = 0
        else:
            self.pairs.append(pair)

    def get_instance_randomly(self, num):
        indexes = []
        for i in range(0, num):
            indexes.append(Random.randint(0, len(self.pairs)))
        results = list()
        results.append(self.pairs[index] for index in indexes)
        return results


