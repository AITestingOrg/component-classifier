
from rnn import GRU
import numpy as np

class Ensemble:
    def __init__(self, word_dim, hidden_dim, output_dim, bptt_truncate= 1):
        GRU.__init__(self, word_dim, hidden_dim, output_dim, 1)
        self.wi = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.wh = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, hidden_dim))
        self.aw = self.wh = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, hidden_dim))
        
    def forward(self, x, q):
        T = len(x) #total number of sentences
        q = np.zeros((T, self.word_dim))
        g = np.zeros((T, self.hidden_dim))
        m = np.zeros((T, self.hidden_dim))
        
        # For GRU layers
        z = np.zeros((T, self.hidden_dim))
        r = np.zeros((T, self.hidden_dim))
        
        #holder for states
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        
        o = np.zeros((T, self.output_dim))
                
        for t in np.arange(T):
            # Input module
            z[t] = self.sigmoid(self.zi.dot(x[t]) + self.zh.dot(s[t - 1]))
            r[t] = self.sigmoid(self.ri.dot(x[t]) + self.rh.dot(s[t - 1]))
            s[t] = np.multiply(z[t], s[t - 1]) + np.multiply((1 - z[t]), np.tanh(self.U.dot(x[t]) + self.W.dot(s[t - 1] * r[t])))
            
            # Episodic memory module
            q[t] = self.calc_gru(q, s) # Question module
            g[t] = self.calc_attention(x, q, m)
            m[t] = np.multiply(g[t], s[-2]) + np.multiply((1 - g[t]), s[-2])
            
            # Answer module
            a[t] = self.calc_gru([o[t - 1], q])
            o[t] = self.softmax(self.aw.dot(a[t]))
        return o, s
        
    def calc_gru(self, x, s):
        z = self.sigmoid(self.zi.dot(x) + self.zh.dot(s[-2]))
        r = self.sigmoid(self.ri.dot(x) + self.rh.dot(s[-2]))
        s = np.multiply(z, s[-2]) + np.multiply((1 - z), np.tanh(self.U.dot(x) + self.W.dot(s[-2] * r)))
        return s
        
    def calc_attention(self, x, q, m):
        # Calculate G for episodic memory 
        T = len(x)
        q = self.forward(q)
        s = self.forward(x)
        
        for t in np.arange(T):
            z[t] = np.array([np.multiply(s, q), np.multiply(s, m[t - 1]), np.absolute(s - q), np.absolute(s - m[t - 1])])
            Z[t] = self.wi.dot(np.tanh(self.wh.dot(z[t])))
            g[t] = self.softmax(Z[t])
        return g
        
    def calc_episodic_hidden(self, x, q, m):
        # Calculate hiddenstate of episodic memory
        m = np.multiply(g, s[-2]) + np.multiply((1 - g), s[-2])
        return m
        
    def answer(self, y, q):
        x = [y[-2], q]
        o = self.forward(x)
        return self.softmax(self.aw.dot(o))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
        
            