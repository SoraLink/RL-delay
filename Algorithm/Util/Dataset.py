from random import Random
class Dataset():
    def __init__(self, max_size):
        self.max_size = max_size
        self.pairs = []
        self.rewards = []
        self.index = 0

    def add_instance(self, pair, reward):
        if len(self.pairs) >= self.max_size:
            self.pairs[self.index] = pair
            self.rewards[self.index] = reward
            if self.index < self.max_size:
                self.index += 1
            else:
                self.index = 0
        else:
            self.pairs.append(pair)
            self.rewards.append(reward)

    def get_instance_randomly(self, num):
        indexes = []
        for i in range (0,num):
            indexes.append(Random.randint(0,len(self.pairs)))
        results = [[],[]]
        results[0].append(self.pairs[index] for index in indexes )
        results[1].append(self.rewards[index] for index in indexes)
        return results


