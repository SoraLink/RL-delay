import random
class Dataset():
    def __init__(self, max_size, ):
        self.max_size = max_size
        self.pairs = []
        self.index = 0

    def add_instance(self, pair):
        if len(self.pairs) >= self.max_size:
            if self.index < self.max_size:
                self.index += 1
            else:
                self.index = 0
            self.pairs[self.index] = pair
        else:
            self.pairs.append(pair)

    def get_instance_randomly(self, num):
        indexes = []
        for i in range(0, num):
            indexes.append(random.randint(0, len(self.pairs)-1))
        
        results = [self.pairs[i] for i in indexes]
        return results


