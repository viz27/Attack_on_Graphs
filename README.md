# Attack on Graphs

Project analyzing effectiveness of adversarial attack methods on Graph Neural Networks and defense methods. Two papers are currently considered in this project. [Nettatack](https://arxiv.org/abs/1805.07984) and [GCNJaccard](https://arxiv.org/abs/1903.01610). All experiments are performed with implementations of these methods available at [Deeprobust](https://github.com/DSE-MSU/DeepRobust)- which is a pytorch adversarial library for attack and defense methods on images and graphs. More details are available in the [report](https://github.com/viz27/Attack_on_Graphs/blob/master/Term_Paper.pdf).

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

## Experiment 1 - Jaccard score threshold’s impact on clean graph while running GCNJaccard
Run as given below, specifying the dataset and Jaccar Score Threshold.
```
python GCN_vs_GCNJaccar.py --dataset cora --threshold 0.01
```
| Threshold       | Edges Removed(Citeseer) | Accuracy(Citeseer) | Edges Removed(Cora) | Accuracy(Cora) |
|-------|-------|-------|-------|-------|
| 0.00 | 0 | 0.7198 | 0 | 0.8234 
| 0.01 | 96 | 0.7216 | 548 | 0.8234
| 0.02 | 360 | 0.7174 | 548 | 0.8234
| 0.03 | 483 | 0.7186 | 1015 | 0.8104
| 0.04 | 817 | 0.718 | 1203 | 0.7953
| 0.05 | 1085 | 0.7079 | 1561 | 0.7741
| 0.06 | 1385 | 0.715 | 2046 | 0.7596
| 0.07 | 1658 | 0.7139 | 2248 | 0.747
| 0.08 | 1912 | 0.6991 | 2692 | 0.7339
| 0.09 | 2121 | 0.6985 | 3032 | 0.7208
| 0.10 | 2284 | 0.6949 | 3253 | 0.7133

## Experiment 2 - Effectiveness of NETTACK and GCNJaccard(Against NETTACK)
Run as given below, specifying the dataset and Jaccar Score Threshold.
```
python nettack_and_defense.py --dataset cora --threshold 0.04
```
| Method | Accuracy(Citeseer) | Accuracy(Cora) |
|-------|-------|-------|
NETTACK |	0.1 | ***0.15***
GCNJaccard(0.01) | 0.3 | 0.3
GCNJaccard(0.02) | 0.525 | 0.3
GCNJaccard(0.03) | 0.675 | 0.4
GCNJaccard(0.04) | ***0.7*** | 0.45
GCNJaccard(0.05) | 0.65 | *0.55*
GCNJaccard(0.10) | 0.775 | 0.625
