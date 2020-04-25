import numpy as np

def get_embedding_mat(words, embedding):
    mat = []
    for word in words:
        mat.append(embedding[word])
    return np.asarray(mat)