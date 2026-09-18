import sklearn
import numpy as np
from my_co_matrix import create_co_matrix, into_corpus
from sklearn.decomposition import TruncatedSVD
class CBOW():
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.W_in = np.random.randn(vocab_size, hidden_size)
        self.W_out = np.random.randn(hidden_size, vocab_size)
    def one_hot(self,context):
        one_hot_vectors = np.zeros((len(context), self.vocab_size))
        for i, word_id in enumerate(context):
            one_hot_vectors[i, word_id] = 1
        return one_hot_vectors
    def forward(self,context):
        pass
    def backward(self, dL_dy):
        pass