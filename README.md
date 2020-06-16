# Attack on Graphs

Project analyzing effectiveness of adversarial attack methods on Graph Neural Networks and defense methods. Two papers are currently considered in this project. [Nettatack](https://arxiv.org/abs/1805.07984) and [GCNJaccard](https://arxiv.org/abs/1903.01610). All experiments are performed with implementations of these methods available at [Deeprobust](https://github.com/DSE-MSU/DeepRobust)- which is a pytorch adversarial library for attack and defense methods on images and graphs.

## Installation
Install DeepRobust library as
```
git clone https://github.com/DSE-MSU/DeepRobust.git
python DeepRobust/setup.py install
```
Then checkout this project's code as
```
git clone https://github.com/viz27/Attack_on_Graphs.git
cd Attack_on_Graphs
```

## Experiment 1 - Effect of Jaccard Similarity threshold on overall GCN accuracy
### Citeseer
| Threshold       | Edges Removed | Accuracy |
|-------|-------|-------|
| 0.00 | 0 | 0.7198 |
| 0.01 | 96 | 0.7216
| 0.02 | 360 | 0.7174
| 0.03 | 483 | 0.7186
| 0.04 | 817 | 0.718
| 0.05 | 1085 | 0.7079
| 0.06 | 1385 | 0.715
| 0.07 | 1658 | 0.7139
| 0.08 | 1912 | 0.6991
| 0.09 | 2121 | 0.6985
| 0.10 | 2284 | 0.6949

### Cora
| Threshold        | Edges Removed | Accuracy |
|-------|-------|-------|
0.00 | 0 | 0.8234
0.01 | 548 | 0.8234
0.02 | 548 | 0.8234
0.03 | 1015 | 0.8104
0.04 | 1203 | 0.7953
0.05 | 1561 | 0.7741
0.06 | 2046 | 0.7596
0.07 | 2248 | 0.747
0.08 | 2692 | 0.7339
0.09 | 3032 | 0.7208
0.10 | 3253 | 0.7133
