import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv, GINConv
from models.utils import get_positive_expectation, get_negative_expectation
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from IPython import embed
import numpy as np


class RGCN(nn.Module):
    def __init__(self, g, input_dim, hidden_dim,
                 final_dropout, graph_pooling_type,):

        super(RGCN, self).__init__()
        self.g = g

        # List of gnns
        self.gnn_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.gnn_layers.append(RelGraphConv(in_feat=input_dim,
                                           out_feat=hidden_dim,
                                           num_rels=5,
                                           activation=F.leaky_relu,
                                           dropout=0.3))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.gnn_layers.append(RelGraphConv(in_feat=input_dim,
                                           out_feat=hidden_dim,
                                           num_rels=5,
                                           activation=F.leaky_relu,
                                           dropout=0.3))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.drop = nn.Dropout(final_dropout)
        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, h):
        # list of hidden representation at each layer (including input)
        etypes = self.g.edata['etype']
        # print(etypes.shape)
        hidden_rep = [h]
        for i in range(2):
            h = self.gnn_layers[i](g=self.g, feat=h, etypes=etypes)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        return h


class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()        
        self.g = g
        self.conv = RGCN(g=g, input_dim=in_feats, hidden_dim=n_hidden,
                         final_dropout=dropout, graph_pooling_type='sum')

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features


class GNNDiscLayer(nn.Module):
    def __init__(self, in_feats, n_hidden):
        super(GNNDiscLayer, self).__init__()
        self.fc = nn.Linear(in_feats, n_hidden)
        self.layer_1 = True

    def reduce(self, nodes):
        return {'m': F.relu(self.fc(nodes.data['x']) + nodes.mailbox['m'].mean(dim=1) ), 'root': nodes.mailbox['root'].mean(dim=1)}

    def msg(self, edges):
        if self.layer_1:
            return {'m': self.fc(edges.src['x']), 'root': edges.src['root']}
        else:
            return {'m': self.fc(edges.src['m']), 'root': edges.src['root']}
    
    def concat_edges(self, edges):
        return {'output':torch.cat([edges.src['root'], edges.src['m'], edges.dst['x']], dim=1)}

    def forward(self, g, v, edges, depth=1):
        if depth == 1:
            self.layer_1 = True
        else:
            self.layer_1 = False
        g.apply_edges(self.concat_edges, edges)
        g.push(v, self.msg, self.reduce)
        
        return g.edata.pop('output')[edges]


class SubGDiscriminator(nn.Module):
    def __init__(self, g, in_feats, n_hidden, model_id, n_layers = 2):
        super(SubGDiscriminator, self).__init__()
        self.g = g
        # in_feats = in_feats
        self.dc_layers = nn.ModuleList()
        for i in range(n_layers):
            if model_id > 0:
                self.dc_layers.append(GNNDiscLayer(in_feats, n_hidden))
        
        self.linear = nn.Linear(in_feats + 2 * n_hidden, n_hidden, bias = True)
        self.in_feats = in_feats
        self.model_id = model_id
        self.U_s = nn.Linear(n_hidden, 1)

    def edge_output(self, edges):
        if self.model_id == 1:
            return {'h': torch.cat([edges.src['root'], edges.dst['x']], dim=1)}
        elif self.model_id in [2,3]:
            return {'h': torch.cat([edges.src['root'], edges.src['m'], edges.dst['x']], dim=1)}

    def forward(self, ego_graph, emb, features):
        ego_graph.ndata['root'] = emb[ego_graph.ndata['_ID']]
        ego_graph.ndata['x'] = features[ego_graph.ndata['_ID']]
        ego_graph.ndata['m']= torch.zeros_like(emb[ego_graph.ndata['_ID']])
        edge_embs = []

        for i in ego_graph.nodes():
            v = i
            uid = ego_graph.out_edges(v, 'eid')

            if i+1 == 2:
                h = self.dc_layers[0](ego_graph, v, uid, 1)
            else:
                h = self.dc_layers[0](ego_graph, v, uid, 2)

            edge_embs.append(self.U_s(F.relu(self.linear(h))))
        return edge_embs


class REGI(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, model_id=0, pretrain=None):
        super(REGI, self).__init__()

        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
        self.g = g

        self.subg_disc = SubGDiscriminator(g, in_feats, n_hidden, model_id)
        self.loss = nn.BCEWithLogitsLoss()
        self.model_id = model_id
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        if pretrain is not None:
            print("Loaded pre-train model: {}".format(pretrain) )
            self.load_state_dict(torch.load(pretrain))
    
    def reset_parameters(self):
        self.encoder = Encoder(self.g, self.in_feats, self.n_hidden, self.n_layers, self.activation, self.dropout)
        self.encoder.conv.g = self.g
        self.subg_disc = SubGDiscriminator(self.g, self.in_feats, self.n_hidden, self.model_id)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, ego_graph, features):
        positive = self.encoder(features, corrupt=False)
        perm = torch.randperm(self.g.number_of_nodes())
        negative = positive[perm]
        
        positive_batch = self.subg_disc(ego_graph, positive, features)
        negative_batch = self.subg_disc(ego_graph, negative, features)
        E_pos, E_neg, l = 0.0, 0.0, 0.0
        pos_num, neg_num = 0, 0
        for positive_edge, negative_edge in zip(positive_batch, negative_batch):
            E_pos += get_positive_expectation(positive_edge, 'JSD', average=False).sum()
            pos_num += positive_edge.shape[0]
            E_neg += get_negative_expectation(negative_edge, 'JSD', average=False).sum()
            neg_num += negative_edge.shape[0]
            l += E_neg - E_pos
        return E_neg / neg_num - E_pos / pos_num