import numpy as np
from .utils import get_embedding_mat

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
        
        u, s, vh = np.linalg.svd(self._C)
        B = vh[:self._k, :].T
        return B

    def debiasing(self, es):
        if self._method == 'Hard':
            debiased_embeddings = []
            for _, mat in enumerate(es):
                neutralized_mat = np.zeros_like(mat)
                projected_mat = np.zeros_like(mat)
                debiased_mat = np.zeros_like(mat)
                for j in range(mat.shape[0]):
                    neutralized_mat[j], projected_mat[j] = self._neutralize_word(mat[j])
                mu = np.mean(neutralized_mat, axis=0)
                mu_b = self._Qb @ mu
                v = mu - mu_b
                for j in range(mat.shape[0]):
                    wb_mub = projected_mat[j] - mu_b
                    debiased_mat[j] = v + np.sqrt(1 - np.linalg.norm(v)**2) * wb_mub / np.linalg.norm(wb_mub)
                debiased_embeddings.append(debiased_mat)
            return self._B, debiased_embeddings
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
