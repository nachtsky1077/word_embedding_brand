import numpy as np

class WEAT(object):
    """
    Perform WEAT (Word Embedding Association Test) bias tests on a language model.
    Follows from Caliskan et al 2017 (10.1126/science.aal4230).
    Code from https://gist.github.com/SandyRogers/e5c2e938502a75dcae25216e4fae2da5.js
    """
    
    def __init__(self, model):
        self.model = model

    @staticmethod
    def cos(w1, w2):
        return w1.dot(w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))

    @staticmethod
    def word_association_with_attribute(w, A, B):
        return np.mean([cos(w, a) for a in A]) - np.mean([cos(w, b) for b in B])

    @staticmethod
    def differential_assoication(X, Y, A, B):
        return np.sum([word_association_with_attribute(x, A, B) for x in X]) - np.sum([word_association_with_attribute(y, A, B) for y in Y])

    @staticmethod
    def weat_effect_size(X, Y, A, B):
        return (
            np.mean([word_association_with_attribute(x, A, B) for x in X]) -
            np.mean([word_association_with_attribute(y, A, B) for y in Y])
        ) / np.std([word_association_with_attribute(w, A, B) for w in X + Y])

    @staticmethod
    def random_permutation(iterable, r=None):
        pool = tuple(iterable)
        r = len(pool) if r is None else r
        return tuple(random.sample(pool, r))

    @staticmethod
    def weat_p_value(X, Y, A, B, sample):
        size_of_permutation = min(len(X), len(Y))
        X_Y = X + Y
        observed_test_stats_over_permutations = []

        permutations = [random_permutation(X_Y, size_of_permutation) for s in range(sample)]

        for Xi in permutations:
            Yi = filterfalse(lambda w: w in Xi, X_Y)
            observed_test_stats_over_permutations.append(differential_assoication(Xi, Yi, A, B))

        unperturbed = differential_assoication(X, Y, A, B)
        is_over = np.array([o > unperturbed for o in observed_test_stats_over_permutations])
        return is_over.sum() / is_over.size

    @staticmethod
    def weat_stats(X, Y, A, B, sample_p=None):
        test_statistic = differential_assoication(X, Y, A, B)
        effect_size = weat_effect_size(X, Y, A, B)
        p = weat_p_value(X, Y, A, B, sample=sample_p)
        return test_statistic, effect_size, p

    def run_test(self, target_1, target_2, attributes_1, attributes_2, sample_p):
        """Run the WEAT test for differential association between two 
        sets of target words and two seats of attributes.
        
        EXAMPLE:
            >>> test.run_test(a, b, c, d, sample_p=1000) # use 1000 permutations for p-value calculation
            >>> test.run_test(a, b, c, d, sample_p=None) # use all possible permutations for p-value calculation
            
        RETURNS:
            (d, e, p). A tuple of floats, where d is the WEAT Test statistic, 
            e is the effect size, and p is the one-sided p-value measuring the
            (un)likeliness of the null hypothesis (which is that there is no
            difference in association between the two target word sets and
            the attributes).
            
            If e is large and p small, then differences in the model between 
            the attribute word sets match differences between the targets.
        """
        X = [self.model(w) for w in target_1]
        Y = [self.model(w) for w in target_2]
        A = [self.model(w) for w in attributes_1]
        B = [self.model(w) for w in attributes_2]
        return weat_stats(X, Y, A, B, sample_p)

