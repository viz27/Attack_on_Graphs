import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense import GCNJaccard
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--threshold', type=float, default=0.01,  help='Threshold Jaccard Similarity')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
print(adj.shape)
print(features.shape)
print(labels.shape)
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)


def display_graph_stats():
    print("Dataset :", args.dataset)
    print("Nodes :", adj.shape[0])
    print("Edges :", int(adj.sum()/2))


def test_GCN():
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train)

    gcn.eval()
    output = gcn.predict()
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("GCN Test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def test_GCNJaccard():
    ''' test on GCNJaccard (poisoning attack)'''
    model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=16, device=device)

    model = model.to(device)

    model.fit(features, adj, labels, idx_train, threshold=args.threshold)

    model.eval()
    output = model.predict()
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("GCNJaccard Test set results:",
          "accuracy = {:.4f}, threshold = {:.2f}".format(acc_test.item(), args.threshold))
    return acc_test.item()

def main():
    display_graph_stats()
    test_GCN()
    test_GCNJaccard()


if __name__ == '__main__':
    main()

