import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

class GRU:
    def __init__(self, word_dim, hidden_dim, output_dim, bptt_truncate= 1):
        np.random.seed(10)
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bptt_truncate = bptt_truncate
        
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (output_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        
        # Z weights
        self.zi = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.zh = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, hidden_dim))
        
        # R weights
        self.ri = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.rh = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, hidden_dim))
        
    def setUpWeights(self, U, V, W):
        self.U = U
        self.V = V
        self.W = W
    
    def forward_propagation(self, x):
        T = len(x) #total amount of time steps
        
        # Fro GRU layers
        z = np.zeros((T, self.hidden_dim))
        r = np.zeros((T, self.hidden_dim))
        
        #holder for states
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        
        o = np.zeros((T, self.output_dim))
        
        #save state and output for each timestep
        for t in np.arange(T):
            #gates
            z[t] = self.sigmoid(self.zi.dot(x[t]) + self.zh.dot(s[t - 1]))
            r[t] = self.sigmoid(self.ri.dot(x[t]) + self.rh.dot(s[t - 1]))
            s[t] = z[t] * s[t - 1] + (1 - z[t]) * (np.tanh(self.U.dot(x[t]) + self.W.dot(s[t - 1] * r[t])))
            o[t] = self.softmax(self.V.dot(s[t]))
            # s[t] = np.tanh(self.U.dot(x[t]) + self.W.dot(s[t - 1]))
            # o[t] = self.softmax(self.V.dot(s[t]))
        return [o, s]
        
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
        
    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        
        dldzi = np.zeros(self.zi.shape)
        dldzh = np.zeros(self.zh.shape)
        dldri = np.zeros(self.ri.shape)
        dldrh = np.zeros(self.rh.shape)
        
        delta_o = o - y

        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (s[t] - s[t] ** 2)
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dldzh += np.outer(delta_t, s[bptt_step - 1])
                dldrh += np.outer(delta_t, s[bptt_step - 1])
                
                dLdU[:,bptt_step] += delta_t
                dldzi[:,bptt_step] += delta_t
                dldri[:,bptt_step] += delta_t
                
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (s[bptt_step - 1] - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW, dldzi, dldzh, dldri, dldrh]
        
    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW, dldzi, dldzh, dldri, dldrh = self.bptt(x, y)
        
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
        
        self.zi -= learning_rate * dldzi
        self.zh -= learning_rate * dldzh
        self.ri -= learning_rate * dldri
        self.rh -= learning_rate * dldrh
        
    def saveWeights(self):
        np.savetxt("U.csv", self.U, delimiter=",")
        np.savetxt("V.csv", self.V, delimiter=",")
        np.savetxt("W.csv", self.W, delimiter=",")    
            