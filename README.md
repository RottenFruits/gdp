# gdp

gdp is generating distributed representation code sets written by pytorch. 

This code sets is including skip gram and cbow.

---
## Installation
### Dependencies

gdp requires:
- python 3.6.6
- pytorch-cpu 1.0.0
- numpy 1.15.4
- tqdm 4.28.1

### User installation

You can install gdp running the following commands.

```
pip install git+https://github.com/RottenFruits/gdp
```

## Example
### skip gram

This is example that run simple skip gram.


```python
from gdp import distributed_representation as dr
from gdp import corpus as cp

data = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]

corpus = cp.Corpus(data = data, mode = "a", max_vocabulary_size = 5000, max_line = 0, 
                   minimum_freq = 0)

window_size = 1
embedding_dims = 30
batch_size = 128

dr_sg = dr.DistributedRepresentation(corpus, embedding_dims, window_size, batch_size, 
                                       model_type = "skip-gram", ns = 0, trace = True)
dr_sg.train(num_epochs = 101, learning_rate = 0.05)
```

### skip gram with negative sampling
If you want to use negative sampling is this.

```python
dr_sgns = dr.DistributedRepresentation(corpus, embedding_dims, window_size, batch_size, 
                                       model_type = "skip-gram", ns = 1, negative_samples = 5, trace = True)
dr_sgns.train(num_epochs = 101, learning_rate = 0.05)
```

### Etc
If you want to use cbow architecture, you should replace `model_type` "skip-gram" to "cbow".

And more example code is in example directory, please check it too.

## Distributed representations

gdp inclues:
- skipgram
- skipgram with negative sampling
- cbow
- cbow with negative sampling

