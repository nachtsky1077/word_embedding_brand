import numpy as np
from .utils import get_embedding_mat
from sklearn.decomposition import PCA

class EmbeddingDebias:

    def __init__(self, ds, embedding, k=1, method='Hard'):
        '''
        ds: sets of directions
        k: subspace dimension
        method: debias model, has to be "Hard" or "Soft"
        '''
        assert method in ['Hard', 'Soft'], "Debiasing method has to be Hard-debiasing or Soft-debiasing."
        self._method = method
        self._Ds = ds
        self._embedding = embedding
        self._k = k
        self._B = self._find_subspace()
        self._Qb = self._B @ self._B.T

    def _find_subspace(self):
        mus = []
        self._w = []
        self._C = []
        for _, mat in enumerate(self._Ds):
            self._w.append(mat)
            mu = np.mean(mat, axis=0)
            mus.append(mu)
            num_d = mat.shape[0]
            for j in range(mat.shape[0]):
                if mat.shape[0] > 1:
                    w_mu = mat[j] - mu
                else:
                    w_mu = mat[j]
                self._C.append(w_mu)
        self._C = np.asarray(self._C)
        
        pca = PCA(n_components=self._k)
        return pca.fit(self._C).components_.T


    def debiasing(self, words, eq_sets):
        if self._method == 'Hard':
            debiased_embeddings = []
            
            # first get projections onto bias subpace
            for i, word in enumerate(words):
                v = self._embedding[word]
                print(self._B.shape)
                v_b = self._Qb @ v
                new_v = (v - v_b) / np.linalg.norm(v - v_b)
                debiased_embeddings.append(new_v)
            
            # neutralize for each equality set
            neutralized_embeddings = dict()
            for eq_set in eq_sets:
                mean = np.zeros((self._Qb.shape[0], ))
                for word in eq_set:
                    mean += self._embedding[word]
                mean /= float(len(eq_set))
                mean_b = self._Qb @ mean
                upsilon = mean - mean_b

                for word in eq_set:
                    v = self._embedding[word]
                    v_b = self._Qb @ v

                    frac = (v_b - mean_b) / np.linalg.norm(v_b - mean_b)
                    new_v = upsilon + np.sqrt(1 - np.sum(np.square(upsilon))) * frac
                    neutralized_embeddings[word] = new_v
            
            return self._B, debiased_embeddings, neutralized_embeddings
        else:
            raise NotImplementedError

    def _neutralize_word(self, w):
        wb = self._Qb @ w
        w = w - wb
        return w / np.linalg.norm(w), wb

    def project(self, word):
        '''
        :word: the word to be projected to the subspace
        return: word projection onto the subspace
        '''
        try:
            word_embedding = self._embedding[word]
        except Exception:
            print('Word "{}" not found in the w2v model.'.format(word))
            return None
        
        return self._Qb @ word_embedding
