# name-disambiguation
this is a pipeline of author name disambiguation.

## Basic requirements

* python 3.6.5
* networkx 1.11
* gensim 3.4.0
* sklearn 0.20.1
* numpy 1.14.3
* pandas 0.23.0
* tensorflow 1.10.0

Note: you are recommended to run this pipeline on windows.

## Run this pipeline
```bash
# step 1: preprocess the data
python data_processing.py

# step 2: train the GRU based encoder to learn deep semantic representations
python DRLgru.py 

# step 3: construct a PHNet and generate random walks
python walks.py

# step 4: weighted heterogeneous network embedding
python WHNE.py

# step 5: generate clustering results
python evaluator.py
```


## Data

 you are recommended to use the word2vec model we pre-trained to generate word embeddings of publication titles via [OneDrive](https://1drv.ms/u/s!AvNheLYVCGGGayqTjhiXoOgRc9w) (or [BaiduYun](https://pan.baidu.com/s/18nTdRcmZ4sKz7RbmrCIfWA)).  Or you can train your own word vectors(dimension = 100) using the [woed2vec method](https://radimrehurek.com/gensim/models/word2vec.html) in gensim library.

