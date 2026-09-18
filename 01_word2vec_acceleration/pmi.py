import numpy as np
from my_co_matrix import create_co_matrix, into_corpus

def ppmi(C,verbose=False,epse=1e-8):
    M = np.zeros_like(C,dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C,axis=0)
    total = C.shape[0]*C.shape[1]
    cnt = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j]*N/(S[j]*S[i])+epse)
            M[i,j] = max(0,pmi)
            if verbose:
                cnt+=1
                if cnt% (total//100 + 1) == 0:
                    print("\r%.1f%% done" % (100*cnt/total),end="")
    return M

if __name__ == "__main__":
    text = "you say goodbye and I say hello."
    corpus, word_to_id, id_to_word = into_corpus(text)
    C = create_co_matrix(corpus, window_size=1)
    W = ppmi(C, verbose=True)
    print(W)