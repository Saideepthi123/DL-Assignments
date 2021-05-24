"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        
        self.W = None  # TODO: change this
        self.lr = lr
        self.num_iters = epochs
        self.alpha = reg_const
        self.n_class = n_class

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        num_train , dim = X.shape
        num_classes = np.max(y) +1
        
        if self.W is None:
                self.W = np.random.randn(dim, num_classes) * 0.001

        loss = 0.0
        
        num_classes = self.W.shape[1]
        num_train = X.shape[0]
        dW = np.zeros(self.W.shape)  # initialize the gradient as zero
        scores = X.dot(self.W)
        scores -= scores.max()
        scores = np.exp(scores)
        scores_sums = np.sum(scores, axis=1)
        cors = scores[range(num_train), y]
        loss = cors / scores_sums
        loss = -np.sum(np.log(loss))/num_train + self.alpha * np.sum(self.W * self.W)

        # grad
        s = np.divide(scores, scores_sums.reshape(num_train, 1))
        s[range(num_train), y] = - (scores_sums - cors) / scores_sums
        dW = X.T.dot(s)
        dW /= num_train
        dW += 2 * self.alpha * self.W

        return loss, dW
    
    def train(self, X: np.ndarray, y: np.ndarray):
        num_train, dim = X.shape
        # TODO: implement me
        
        loss_history = []
        batch_size = 10
        
        for it in range(self.num_iters):
            batch_ind = np.random.choice(num_train, batch_size)
            X_batch = X[batch_ind]
            y_batch = y[batch_ind]

            # evaluate loss and gradient
            loss, grad = self.calc_gradient(X_batch, y_batch)
            loss_history.append(loss)

            self.W += - self.alpha * grad


        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: implement me
        y_pred = np.zeros(X.shape[0])

        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

