### Brand embedding debiasing project

Current usage:
* Download pre-trained word2vec embeddings(support only word2vec model only);
* Change to root folder (the folder contains subfolders like 'code', 'notes');
* Run command "python -m code.main --$PATH_TO_WORD2VEC_MODEL $OTHER_OPTIONS"

##### Arguments
* --pretrained_file: path to pretrained model
* --method: debiasing method, has to be 'Hard' or 'Soft'
* --subspace_dim: number of eigen-vectors preserved for the bias subspace
* --subspace_words: the sets of words defining the bias subspace
* --neutral_words: the sets of neutral words to be debiased
