import numpy as np
eps = 1e-8
def cos_similarity(x, y):
    nx = x/(np.sqrt(np.sum(x**2)+eps))
    ny = y/(np.sqrt(np.sum(y**2)+eps))
    return np.sum(nx*ny)
if __name__ == "__main__":
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    print("Cosine similarity:", cos_similarity(x, y))