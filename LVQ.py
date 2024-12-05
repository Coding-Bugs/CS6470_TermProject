import numpy as np

""" 
Base gotten from 
https://medium.com/@hasanhammad13/building-a-learning-vector-quantization-lvq1-network-from-scratch-with-python-6931c23ba07
"""
class LVQ1:

    def __init__(self, epochs=100, alpha=0.5):
        self.epochs = epochs
        self.alpha = alpha

    # define a function to find the winning vector
    # by calculating euclidean distance  
    def winner(self, sample):
        
        distance_0 = 0
        distance_1 = 0
        
        for i in range(len(sample)):
        # calculate euclidean distance between the given vector and the 1st prototype
            distance_0 = distance_0 + np.linalg.norm(sample[i] - self.weights[0][i])
            # calculate euclidean distance between the given vector and the 2nd prototype
            distance_1 = distance_1 + np.linalg.norm(sample[i] - self.weights[1][i])

            if distance_0 < distance_1:
                return 0 
            else: 
                return 1

    # define a function to update the winning prototype    
    def update( self, sample, l_sample, winner_class) :
        # if the label of the nearest prototype == the label of the given vector
        # then add alpha
        if l_sample == winner_class:
            for i in range(len(self.weights[0])) :
                self.weights[winner_class][i] = self.weights[winner_class][i] + self.alpha * (sample[i] - self.weights[winner_class][i]) 
        # if the label of the nearest prototype != the label of the given vector
        # then subtract alpha
        else:
            for i in range(len(self.weights[0])) :
                org = self.weights[winner_class][i]
                adjust = sample[i] - self.weights[winner_class][i]
                weighted = self.alpha * adjust
                self.weights[winner_class][i] -= weighted
    
    # reduce learning rate
    def reduce_lr(self, step):
        self.alpha = self.alpha - (step * 0.000001)

    def fit(self, X, Y):
        # Initialize vectors 
        self.weights = []
        prototype1 = np.random.uniform(size=len(X[0]))
        prototype2 = np.random.uniform(size=len(X[0]))
        self.weights.append(prototype1)
        self.weights.append(prototype2)

        # remove the prototypes and labels
        # X = np.delete(X, [0, 1], 0)
        # Y = np.delete(Y, [0, 1], 0)

        for i in range(self.epochs):
            # for each sample
            for j in range(len(X)):
                # take the sample
                sample = X[j]
                # find the nearest class
                winner_class = self.winner(sample)
                # take the label of the given sample
                label = Y[j]
                # update prototypes (self.weights)
                self.update(sample, label, winner_class)
            # reduce learning rate
            self.reduce_lr(i)

    def predict(self, X):
        pred = []
        for sample in X:
            y = self.winner(sample)
            pred.append(y)
        return pred