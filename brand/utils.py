import numpy as np

def get_embedding_mat(words, embedding, mean=False):
    mat = []
    for word in words:
        mat.append(embedding[word])
    mat = np.asarray(mat)
    if not mean:
        return mat
    else:
        return mat, np.mean(mat, axis=0)

def get_word_vector(word, kv, mean=None):
    word_vec = kv[word]
    if mean is not None:
        return word_vec-mean
    else:
        return word_vec