import numpy as np
import networkx as nx
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

from models.dgi import DGI, MultiClassifier
from models.regi import REGI

from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm
import pickle
from collections import defaultdict
from sklearn.manifold import SpectralEmbedding
import argparse
import pandas as pd

from dataset import AmazonDataset, EgoGraphDataset, collate_data

def extract_node_degree(graph, max_degree = 32):
    """one-hot node degree"""
    features = th.zeros([graph.number_of_nodes(), max_degree])
    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degree(i), max_degree-1)] = 1
        except:
            features[i][0] = 1
    return features


##### build graph
def build_amazon_graph(df:pd.DataFrame):
    users = df['user_id']
    items = df['item_id'] + max(users) +1
    rating = df['rating']

    src_ids = []
    dst_ids = []
    etype = []

    for u, i, r in zip(*[users, items, rating]):
        src_ids += [int(u), int(i)]
        dst_ids += [int(i), int(u)]
        etype += [int(r),int(r)]
    graph = dgl.graph((src_ids, dst_ids), num_nodes=max(items)+1)
    graph.edata['etype'] = th.Tensor(etype)
    return graph


def get_model(args, graph, in_feats:int):
    if args.model_type == 0:
        dgi = DGI(graph,
                  in_feats,
                  args.n_hidden,
                  args.n_layers,
                  nn.PReLU(args.n_hidden),
                  args.dropout)
    elif args.model_type == 2:
        dgi = REGI(graph,
                    in_feats,
                    args.n_hidden,
                    args.n_layers,
                    nn.PReLU(args.n_hidden),
                    args.dropout,
                    args.model_id)

    return dgi


def random_train_test_mask(labels, valid_mask = None, train_ratio=0.8):
    train_mask = th.zeros(labels.shape, dtype=th.bool)
    test_mask = th.ones(labels.shape, dtype=th.bool)
    
    num_train = int(labels.shape[0] * train_ratio)
    all_node_index = list(range(labels.shape[0]))
    np.random.shuffle(all_node_index)

    train_mask[all_node_index[:num_train]] = 1
    test_mask[all_node_index[:num_train]] = 0
    if valid_mask is not None:
        train_mask *= valid_mask
        test_mask *= valid_mask
    return train_mask, test_mask


def evaluate(model, embeds, df):
    model.eval()
    dataset = AmazonDataset(df, embeds=embeds)
    loader = th.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)
    
    squared_loss = 0
    total = 0
    with th.no_grad():
        for uid, iid, u_ebm, i_emb, label in loader:
            logits = model(th.cat([u_ebm, i_emb], dim=1))
            preds = F.softmax(logits)
            preds = th.sum(preds*th.Tensor([[0.,0.25,0.5,0.75,1.]]), dim=1)
            squared_loss += th.sum((preds-th.div(label,4.))**2)
            total += len(label)
        
    return math.sqrt(squared_loss/total)


##### Train with source graph
def train_src(args, train_file_path, test_file_path):
    """Prepare source"""
    train_df = pd.read_csv(train_file_path, index_col=0)
    test_df = pd.read_csv(test_file_path, index_col=0)
    train_df['rating'] = train_df['rating']-1
    test_df['rating'] = test_df['rating']-1

    train_graph = build_amazon_graph(train_df)
    train_labels = th.LongTensor(train_df['rating'])
    features = extract_node_degree(train_graph)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        th.cuda.set_device(args.gpu)
        features = features.cuda()
        train_labels = train_labels.cuda()

    n_classes = train_labels.max().item() + 1
    in_feats = features.shape[1]
    dgi = get_model(args, train_graph, in_feats)

    if cuda:
        dgi.cuda()
    dgi_optimizer = th.optim.Adam(dgi.parameters(),
                                        lr=args.dgi_lr,
                                        weight_decay=args.weight_decay)
    cnt_wait = 0
    best = 1e9
    train_graph.ndata['features'] = features




    train_dataset = EgoGraphDataset(graph=train_graph)
    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_data, num_workers=8)

    """Train GNN"""    
    for i in range(args.n_dgi_epochs):
        dgi.train()
        loss = 0.0
        
        # REGI mode      
        for eg in tqdm(train_loader):
            dgi_optimizer.zero_grad()
            l = dgi(eg, features)
            l.backward()
            loss += l
            dgi_optimizer.step()        
    
        if loss < best:
            best = loss
            cnt_wait = 0
            th.save(dgi.state_dict(), 'amazon_regi_classification_{}.pkl'.format(args.model_type))
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    return 


def train_dst(args, train_file_path, test_file_path):
    """Prepare source"""
    train_df = pd.read_csv(train_file_path, index_col=0)
    test_df = pd.read_csv(test_file_path, index_col=0)
    train_df['rating'] = train_df['rating']-1
    test_df['rating'] = test_df['rating']-1
    n = len(train_df)

    train_graph = build_amazon_graph(pd.concat([train_df, test_df]))
    features = extract_node_degree(train_graph)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        th.cuda.set_device(args.gpu)
        features = features.cuda()

    n_classes = 5
    in_feats = features.shape[1]
    dgi = get_model(args, train_graph, in_feats)

    """Train Task"""
    # create classifier model
    classifier = MultiClassifier(args.n_hidden*2, n_classes)
    if cuda:
        print('cuda')
        classifier.cuda()

    classifier_optimizer = th.optim.Adam(classifier.parameters(),
                                            lr=args.classifier_lr,
                                            weight_decay=args.weight_decay)

    dgi.load_state_dict(th.load('amazon_regi_classification_{}.pkl'.format(args.model_type)))

    # extract embeddings
    with th.no_grad():
        if args.model_type == 2:
            embeds = dgi.encoder(features, corrupt=False)
        elif args.model_type == 0:
            embeds = dgi.encoder(features)
    
    embeds = embeds.detach()

    train_dataset = AmazonDataset(train_df, embeds=embeds)
    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)


    for epoch in tqdm(range(args.n_classifier_epochs)):
        classifier.train()
        classifier_optimizer.zero_grad()
        for uid, iid, u_ebm, i_emb, label in train_loader:
            preds = classifier(th.cat([u_ebm, i_emb], dim=1))
            loss = F.nll_loss(preds, label)
            loss.backward()
            classifier_optimizer.step()

    acc = evaluate(classifier, embeds, test_df)
    print("Test Accuracy {:.4f}".format(acc))
    return acc

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DGI')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--dgi-lr", type=float, default=1e-2,
                        help="dgi learning rate")
    parser.add_argument("--classifier-lr", type=float, default=1e-2,
                        help="classifier learning rate")
    parser.add_argument("--n-dgi-epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=0,
                        help="early stop patience condition")
    parser.add_argument("--model", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--model-type", type=int, default=2,
                    help="graph self-loop (default=False)")
    parser.add_argument("--graph-type", type=str, default="DD",
                    help="graph self-loop (default=False)")
    parser.add_argument("--file-path", type=str,
                        help="graph path")
    parser.add_argument("--label-path", type=str,
                        help="label path")
    parser.add_argument("--model-id", type=int, default=2,
                    help="[0, 1, 2, 3]")

    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    src_accs = []
    dst_accs_0 = []
    dst_accs_1 = []
    dst_accs_2 = []
    for i in range(10):
        train_src(args, 'data/book_sub_train.csv', 'data/book_sub_valid.csv')
        dst_acc_0 = train_dst(args, 'data/book_sub_train.csv', 'data/book_sub_valid.csv')
        dst_acc_1 = train_dst(args, 'data/music_sub_train.csv', 'data/music_sub_valid.csv')
        dst_acc_2 = train_dst(args, 'data/video_game_sub_train.csv', 'data/video_game_sub_valid.csv')
        dst_accs_0.append(dst_acc_0)
        dst_accs_1.append(dst_acc_1)
        dst_accs_2.append(dst_acc_2)

    print('dst 0', 'Test Accuracy {:.4f}, std {:.4f}'.format(np.mean(dst_accs_0), np.std(dst_accs_0)) )
    print('dst 1', 'Test Accuracy {:.4f}, std {:.4f}'.format(np.mean(dst_accs_1), np.std(dst_accs_1)) )
    print('dst 2', 'Test Accuracy {:.4f}, std {:.4f}'.format(np.mean(dst_accs_2), np.std(dst_accs_2)) )