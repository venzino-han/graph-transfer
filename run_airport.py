# Experiment 2: role-identification on airport dataset


import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from models.dgi import DGI, MultiClassifier
from models.subgi import SubGI
from IPython import embed
import scipy.sparse as sp
from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm
import pickle
from collections import defaultdict
from sklearn.manifold import SpectralEmbedding

import pandas as pd
from dataset import egograph_extraction


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def spectral_feature(graph, args):
    #create adj matrix --> sepctral embedding (n_hidden)
    A = np.zeros([graph.number_of_nodes(), graph.number_of_nodes()])
    a,b = graph.all_edges()
    
    for id_a, id_b in zip(a.numpy().tolist(), b.numpy().tolist()):
        #OUT.write('0 {} {} 1\n'.format(id_a, id_b))
        A[id_a, id_b] = 1
    embedding = SpectralEmbedding(n_components=args.n_hidden)
    features = torch.FloatTensor(embedding.fit_transform(A))
    return features

def degree_bucketing(graph, args, degree_emb=None, max_degree = 10):
    # node degree --> one-hot feature
    max_degree = args.n_hidden
    features = torch.zeros([graph.number_of_nodes(), max_degree])

    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degree(i), max_degree-1)] = 1
        except:
            features[i][0] = 1
    return features

def createTraining(labels, valid_mask = None, train_ratio=0.8):
    train_mask = torch.zeros(labels.shape, dtype=torch.bool)
    test_mask = torch.ones(labels.shape, dtype=torch.bool)
    
    num_train = int(labels.shape[0] * train_ratio)
    all_node_index = list(range(labels.shape[0]))
    np.random.shuffle(all_node_index)
    #for i in range(len(idx) * train_ratio):
    # embed()
    train_mask[all_node_index[:num_train]] = 1
    test_mask[all_node_index[:num_train]] = 0
    if valid_mask is not None:
        train_mask *= valid_mask
        test_mask *= valid_mask
    return train_mask, test_mask

def output_adj(graph):
    A = np.zeros([graph.number_of_nodes(), graph.number_of_nodes()])
    a,b = graph.all_edges()
    for id_a, id_b in zip(a.numpy().tolist(), b.numpy().tolist()):
        A[id_a, id_b] = 1
    return A

def build_graph(df:pd.DataFrame):
    g = DGLGraph()

    user_ids = df['user_id']
    item_ids = df['item_id']
    ratings = df['rating']

    for u, i in zip(user_ids, item_ids):
        g.add_edges(u, i,)
        g.add_edges(i, u)

    # dgl.add_self_loop(g)

    node_degree = g.in_degrees()
    g.ndata['node_degree'] = node_degree
    
    return g, ratings, user_ids, item_ids

hop=2

# dump the best run
def main(args):
    torch.manual_seed(2)
    test_acc = []

    #read data
    df = pd.read_csv('data/music_sub_train.csv', index_col=0)
    df = df.drop_duplicates()
    df['item_id'] += max(df['user_id'])
    
    valid_df = pd.read_csv('data/music_sub_valid.csv', index_col=0)
    valid_df = valid_df.drop_duplicates()
    valid_df['item_id'] += max(df['user_id'])

    train_g, train_rating, train_user_ids, train_item_ids = build_graph(df)
    
    _, valid_rating, valid_user_ids, valid_item_ids = build_graph(valid_df)
    valid_g, _, _, _ = build_graph(pd.concat([df,valid_df]))

    for runs in tqdm(range(10)):
        labels = torch.LongTensor(train_rating)
        
        # degree_emb = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, [100, args.n_hidden])), requires_grad=False)
        # features = degree_bucketing(g, args, degree_emb)

        # train_mask, test_mask = createTraining(labels, valid_mask)
        # # labels = torch.LongTensor(labels)
        # if hasattr(torch, 'BoolTensor'):
        #     train_mask = torch.BoolTensor(train_mask)
        #     #val_mask = torch.BoolTensor(val_mask)
        #     test_mask = torch.BoolTensor(test_mask)
        # else:
        #     train_mask = torch.ByteTensor(train_mask)
        #     #val_mask = torch.ByteTensor(val_mask)
        #     test_mask = torch.ByteTensor(test_mask)

        # in_feats = features.shape[1]
        features = train_g.ndata['node_degree']
        n_classes = labels.max().item() + 1
        n_edges = train_g.number_of_edges()

        if args.gpu < 0:
            cuda = False
        else:
            cuda = True
            torch.cuda.set_device(args.gpu)
            features = features.cuda()
            labels = labels.cuda()

        # g.readonly()
        dgi = SubGI(train_g,
                    in_feats=hop+1,
                    n_hidden=args.n_hidden,
                    n_layers=args.n_layers,
                    activation=nn.PReLU(args.n_hidden),
                    dropout=args.dropout,
                    modle_id=args.model_id)
        # print(dgi)
        if cuda:
            dgi.cuda()

        dgi_optimizer = torch.optim.Adam(dgi.parameters(),
                                        lr=args.dgi_lr,
                                        weight_decay=args.weight_decay)

        cnt_wait = 0
        best = 1e9
        best_t = 0
        dur = []
        train_g.ndata['features'] = features
        for epoch in range(args.n_dgi_epochs):
            # ego-graph extractor (?)
            # NodeFlow generator
            # train_sampler = dgl.contrib.sampling.NeighborSampler(g, 
            #                         batch_size=256, expand_factor=5,
            #                         neighbor_type='in', num_workers=1,
            #                         add_self_loop=False,
            #                         num_hops=args.n_layers+1, shuffle=True)
            dgi.train()
            if epoch >= 3:
                t0 = time.time()
            
            loss = 0.0

            train_sampler =
            # EGI mode
            for nf in train_sampler:
                dgi_optimizer.zero_grad()
                l = dgi(features, nf)
                l.backward()
                loss += l
                dgi_optimizer.step()

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(dgi.state_dict(), 'best_classification_{}.pkl'.format(args.model_type))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!')
                break

            if epoch >= 3:
                dur.append(time.time() - t0)

            #print("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))

        # create classifier model
        classifier = MultiClassifier(args.n_hidden, n_classes)
        if cuda:
            classifier.cuda()

        classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                                lr=args.classifier_lr,
                                                weight_decay=args.weight_decay)

        # flags used for transfer learning
        if args.data_src != args.data_id:
            pass
        else:
            dgi.load_state_dict(torch.load('best_classification_{}.pkl'.format(args.model_type)))

        with torch.no_grad():
            if args.model_type == 1:
                _, embeds, _ = dgi.forward(features)
            elif args.model_type == 2:
                embeds = dgi.encoder(features, corrupt=False)
            elif args.model_type == 0:
                embeds = dgi.encoder(features)
            else:
                dgi.eval()
                test_sampler = dgl.contrib.sampling.NeighborSampler(g, g.number_of_nodes(), -1,  # 0,
                                                                            neighbor_type='in', num_workers=1,
                                                                            add_self_loop=False,
                                                                            num_hops=args.n_layers + 1, shuffle=False)
                for nf in test_sampler:
                    nf.copy_from_parent()
                    embeds = dgi.encoder(nf, False)
                    print("test flow")

        embeds = embeds.detach()

        dur = []
        for epoch in range(args.n_classifier_epochs):
            classifier.train()
            if epoch >= 3:
                t0 = time.time()

            classifier_optimizer.zero_grad()
            preds = classifier(embeds)
            loss = F.nll_loss(preds[train_mask], labels[train_mask])
            # embed()
            loss.backward()
            classifier_optimizer.step()
            
            if epoch >= 3:
                dur.append(time.time() - t0)
            #acc = evaluate(classifier, embeds, labels, train_mask)
            #acc = evaluate(classifier, embeds, labels, val_mask)
            #print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
            #      "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
            #                                    acc, n_edges / np.mean(dur) / 1000))

        acc = evaluate(classifier, embeds, labels, test_mask)
        
        test_acc.append(acc)
        
    print("Test Accuracy {:.4f}, std {:.4f}".format(np.mean(test_acc), np.std(test_acc)))

if __name__ == '__main__':
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
    parser.add_argument("--n-dgi-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience condition")
    parser.add_argument("--model", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--model-type", type=int, default=2,
                    help="graph self-loop (default=False)")
    parser.add_argument("--graph-type", type=str, default="DD",
                    help="graph self-loop (default=False)")
    parser.add_argument("--data-id", type=str,default='',
                    help="[usa, europe, brazil]")
    parser.add_argument("--data-src", type=str, default='',
                    help="[usa, europe, brazil]")
    parser.add_argument("--file-path", type=str,
                        help="graph path")
    parser.add_argument("--label-path", type=str,
                        help="label path")
    parser.add_argument("--model-id", type=int, default=0,
                    help="[0, 1, 2, 3]")

    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    
    main(args)