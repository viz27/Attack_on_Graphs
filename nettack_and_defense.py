import torch
import numpy as np
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense import GCNJaccard
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
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
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

#Display Graph Statistics
print("Graph :", args.dataset)
print("Nodes :", adj.shape[0])
print("Features per node :", features.shape[1])
print("Edges :", int(adj.sum()/2))
print("Train/Val/Test split : {}/{}/{}".format(len(idx_train), len(idx_val), len(idx_test)))


def select_nodes():
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train)
    gcn.eval()
    output = gcn.predict()

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other

def multi_test():
    # Setup Surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)

    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
    
    attack_cnt = 0
    defense_cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes()
    num = len(node_list)
    print('=== Attacking %s nodes one after another ===' % num)
    for target_node in node_list:
        #n_perturbations = min(15, int(degrees[target_node]))
        n_perturbations = int(degrees[target_node])
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, n_perturbations, verbose=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features
        attack_acc = test_attack(modified_adj, modified_features, target_node)
        if attack_acc == 0:
            attack_cnt += 1
        defense_acc = test_defense(modified_adj, modified_features, target_node)
        if defense_acc == 1:
            defense_cnt += 1
        print("Attacking node :", target_node, "Details :", degrees[target_node], modified_adj.sum(0).A1[target_node], attack_acc, defense_acc)
    print('Pure attack accuracy rate : %s' % ((num - attack_cnt)/num), flush=True)
    print('With Jaccard defense accuracy rate : %s' % (defense_cnt/num), flush=True)
    print("GCNJaccard threshold used :", args.threshold)


def test_attack(adj, features, target_node):
    ''' test on GCN after NETTACK poisoning attack'''
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train)
    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])
    acc_test = accuracy(output[[target_node]], [labels[target_node]])
    return acc_test.item()


def test_defense(adj, features, target_node):
    ''' test on GCNJaccard defense after NETTACK poisoning attack'''
    model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max()+1,
                nhid=16, device=device)
    model = model.to(device)
    model.fit(features, adj, labels, idx_train, threshold=args.threshold, verbose=False)
    model.eval()
    output = model.predict()
    probs = torch.exp(output[[target_node]])
    acc_test = accuracy(output[[target_node]], [labels[target_node]])
    # print("Test set results:", "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


if __name__ == '__main__':
    #main()
    multi_test()

