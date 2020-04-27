from argparse import ArgumentParser
import traceback
import gensim
import numpy as np
from .debiasing import EmbeddingDebias
from .utils import get_embedding_mat

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pretrained_file', action='store', type=str, required=True, help='pretrained word embedding file name')
    parser.add_argument('--method', action='store', type=str, default='Hard', help='debias method')
    parser.add_argument('--subspace_dim', action='store', type=int, default=1, help='number of eigen-vectors preserved for bias subspace')
    parser.add_argument('--subspace_words', action='store', type=str,
                        default='she,he;her,his;woman,man;Mary,John;herself,himself;daughter,son;mother,father;gal,guy;girl,boy;female,male', 
                        help='D sets of words, has to be in the form: w1_d1,w2_d1,...;w1_d2,w2_d2,...;...')
    parser.add_argument('--neutral_words', action='store', default='nurse', help='E sets of words, same format as subspace_words')
    args = parser.parse_args()

    try:
        # load w2v word embeddings
        b_flag = True if 'bin' in args.pretrained_file else False
        kv = gensim.models.KeyedVectors.load_word2vec_format(args.pretrained_file, binary=b_flag, limit=500000)
    except Exception:
        print('Fail to load pretrained word2vec model.')
        traceback.print_exc()
        exit(-1)

    ds = [[word for word in D.split(',')] for D in args.subspace_words.split(';')]
    
    # prepare direction matrix sets
    dmat = []
    for _, words in ds:
        mat = []
        for word in words:
            mat.append(get_embedding_mat(word, kv))
        dmat.append(np.asarray(mat))

    es = [[word for word in E.split(',')] for E in args.neutral_words.split(';')]
    debias_worker = EmbeddingDebias(dmat, embedding=kv, k=args.subspace_dim, method=args.method)
    print('Debiasing worker initialization completed.')
    
    print('Debiasing using {}-debias method for word sets {}'.format(args.method, es))

    subspace, new_embeddings = debias_worker.debiasing(es)    
    print('subspace shape:', subspace.shape)
    for i, e in enumerate(es):
        for j, word in enumerate(e):
            print('Before debiasing:')
            print('{} : {}'.format(word, kv[word]))
            print('After debiasing:')
            print('{} : {}'.format(word, new_embeddings[i][j]))
    