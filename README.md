# gdp

gdp is generating distributed representation code sets written by pytorch.

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

There are example codes in example directory.

You can run simply version which using small data of skipgram with negative sampling following command.
```
cd gdp
python example/example_sgns.py
```

## Distributed representations

gdp inclues:
- skipgram
- skipgram with negative sampling
- cbow
- cbow with negative sampling

