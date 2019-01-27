# name-disambiguation
this is a pipeline of author name disambiguation.

## Requirements

* python 3.6
* networkx
* gensim
* sklearn
* numpy
* pandas
* tensorflow

Note: you are recommended to run this pipeline on windows.

## Run this pipeline
```bash
# step 1: preprocess the data
python data_processing.py

# step 2: train the GRU based encoder to learn deep semantic representations
python DRLgru.py 

# step 3: construct a PHNet and generate random walks
python walks.py

#step 4: weighted heterogeneous network embedding
python WHNE.py

#step 4: generate clustering results
python evaluator.py
```


## Data

 you are recommended to use the [word2vec model](https://1drv.ms/u/s!AvNheLYVCGGGayqTjhiXoOgRc9w) we pre-trained. Or you can train your own word vectors(dimension = 100) using the [woed2vec method](https://radimrehurek.com/gensim/models/word2vec.html) in gensim library.

