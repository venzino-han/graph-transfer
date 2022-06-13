import torch as th
import pandas as pd
import numpy as np
import dgl

from bipartite_graph import BipartiteGraph


#######################
# user-item Subgraph Extraction 
#######################

def map_newid(df, col):
    old_ids = df[col]
    old_id_uniq = old_ids.unique()

    id_dict = {old: new for new, old in enumerate(sorted(old_id_uniq))}
    new_ids = np.array([id_dict[x] for x in old_ids])

    return new_ids


def one_hot(idx, length):
    x = th.zeros([len(idx), length], dtype=th.int32)
    x[th.arange(len(idx)), idx] = 1.0
    return x


def get_neighbor_nodes_labels(u_node_idx, i_node_idx, graph, 
                              hop=1, sample_ratio=1.0, max_nodes_per_hop=200):

    # 1. neighbor nodes sampling
    dist = 0
    u_nodes, i_nodes = th.unsqueeze(u_node_idx, 0), th.unsqueeze(i_node_idx, 0)
    u_dist, i_dist = th.tensor([0], dtype=th.long), th.tensor([0], dtype=th.long)
    u_visited, i_visited = th.unique(u_nodes), th.unique(i_nodes)
    u_fringe, i_fringe = th.unique(u_nodes), th.unique(i_nodes)

    for dist in range(1, hop+1):
        # sample neigh alternately
        # diff from original code : only use one-way edge (u-->i)
        u_fringe, i_fringe = graph.in_edges(i_fringe)[0], graph.out_edges(u_fringe)[1]
        u_fringe = th.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
        i_fringe = th.from_numpy(np.setdiff1d(i_fringe.numpy(), i_visited.numpy()))
        u_visited = th.unique(th.cat([u_visited, u_fringe]))
        i_visited = th.unique(th.cat([i_visited, i_fringe]))

        if sample_ratio < 1.0:
            shuffled_idx = th.randperm(len(u_fringe))
            u_fringe = u_fringe[shuffled_idx[:int(sample_ratio*len(u_fringe))]]
            shuffled_idx = th.randperm(len(i_fringe))
            i_fringe = i_fringe[shuffled_idx[:int(sample_ratio*len(i_fringe))]]
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                shuffled_idx = th.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
            if max_nodes_per_hop < len(i_fringe):
                shuffled_idx = th.randperm(len(i_fringe))
                i_fringe = i_fringe[shuffled_idx[:max_nodes_per_hop]]
        if len(u_fringe) == 0 and len(i_fringe) == 0:
            break

        u_nodes = th.cat([u_nodes, u_fringe])
        i_nodes = th.cat([i_nodes, i_fringe])
        u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist,)])
        i_dist = th.cat([i_dist, th.full((len(i_fringe), ), dist,)])

    nodes = th.cat([u_nodes, i_nodes])

    # 2. node labeling
    # this labeling is based on hop from starting nodes
    u_node_labels = th.stack([x*2 for x in u_dist])
    v_node_labels = th.stack([x*2+1 for x in i_dist])
    node_labels = th.cat([u_node_labels, v_node_labels])

    return nodes, node_labels


def subgraph_extraction_labeling(u_node_idx, i_node_idx, graph,
                                 hop=1, sample_ratio=1.0, max_nodes_per_hop=200,):

    # extract the h-hop enclosing subgraph nodes around link 'ind'
    nodes, node_labels = get_neighbor_nodes_labels(u_node_idx=u_node_idx, i_node_idx=i_node_idx, graph=graph, 
                                                  hop=hop, sample_ratio=sample_ratio, max_nodes_per_hop=max_nodes_per_hop)

    subgraph = dgl.node_subgraph(graph, nodes, store_ids=True) 

    subgraph.ndata['nlabel'] = one_hot(node_labels, (hop+1)*2)
    subgraph.ndata['x'] = subgraph.ndata['nlabel']

    # set edge mask to zero as to remove links between target nodes in training process
    subgraph.edata['edge_mask'] = th.ones(subgraph.number_of_edges(), dtype=th.int32)
    su = subgraph.nodes()[subgraph.ndata[dgl.NID]==u_node_idx]
    si = subgraph.nodes()[subgraph.ndata[dgl.NID]==i_node_idx]
    _, _, target_edges = subgraph.edge_ids([su, si], [si, su], return_uv=True)
    subgraph.edata['edge_mask'][target_edges.to(th.long)] = 0
    
    # mask target edge label
    subgraph.edata['label'][target_edges.to(th.long)] = 0.0

    # timestamp normalization
    # compute ts diff from target edge & min-max normalization
    n = subgraph.edata['ts'].shape[0]
    timestamps = subgraph.edata['ts'][:n//2]
    standard_ts = timestamps[target_edges.to(th.long)[0]]
    timestamps = th.abs(timestamps - standard_ts.item())
    timestamps = 1 - (timestamps - th.min(timestamps)) / (th.max(timestamps)-th.min(timestamps) + 1e-5)
    subgraph.edata['ts'] = th.cat([timestamps, timestamps], dim=0) + 1e-5

    return subgraph    

#######################
# Ego-graph Extraction 
#######################
def get_egograph_neighbor(center_node_idx:int, graph:dgl.DGLGraph, 
                          hop=1, max_nodes_per_hop=20):

    # 1. neighbor nodes sampling
    node_dist = th.tensor([0], dtype=th.long)
    visited_nodes = th.tensor([center_node_idx], dtype=th.long)
    nodes = th.tensor([center_node_idx], dtype=th.long)
    fringe = th.tensor([center_node_idx], dtype=th.long)

    for dist in range(1, hop+1):
        fringe = graph.in_edges(fringe)[0]
        fringe = th.from_numpy(np.setdiff1d(fringe.numpy(), visited_nodes.numpy()))
        visited_nodes = th.unique(th.cat([visited_nodes, fringe]))

        if max_nodes_per_hop < len(fringe):
            shuffled_idx = th.randperm(len(fringe))
            fringe = fringe[shuffled_idx[:max_nodes_per_hop]]
        
        if len(fringe) == 0 :
            break
        
        nodes = th.cat([nodes, fringe])
        node_dist = th.cat([node_dist, th.full((len(fringe),), dist,)])
        
    # 2. node labeling
    # this labeling is based on hop from starting nodes
    node_labels = th.stack([x for x in node_dist])

    return nodes, node_labels



def egograph_extraction(node_idx, graph,
                        hop=1, max_nodes_per_hop=10,):

    # extract the h-hop enclosing subgraph nodes around link 'ind'
    nodes, node_labels = get_egograph_neighbor(center_node_idx=node_idx, graph=graph, 
                                               hop=hop, max_nodes_per_hop=max_nodes_per_hop)

    subgraph = dgl.node_subgraph(graph, nodes, store_ids=True) 

    subgraph.ndata['nlabel'] = one_hot(node_labels, hop+1)
    subgraph.ndata['x'] = subgraph.ndata['nlabel']
    return subgraph    



#######################
# Ego graph Dataset 
#######################

class EgoGraphDataset(th.utils.data.Dataset):
    def __init__(self, graph, 
                hop=2, max_nodes_per_hop=5):

        self.nodes = graph.nodes()
        self.graph = graph

        self.hop = hop
        self.max_nodes_per_hop = max_nodes_per_hop

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        node_idx = self.nodes[idx]
        ego_graph = egograph_extraction(node_idx, self.graph, hop=self.hop, max_nodes_per_hop=self.max_nodes_per_hop)
        
        return ego_graph


def collate_data(data):
    g_list = data
    g = dgl.batch(g_list)
    return g


""" Dataset for classifier"""
class AmazonDataset(th.utils.data.Dataset):
    def __init__(self, df:pd.DataFrame, embeds, start=0, end=-1):
        df = df.reset_index()
        self.labels = df['rating'] #pre processed 0~4
        self.users = df['user_id']
        self.items = df['item_id'] + max(self.users) + 1
        self.embeds = embeds[start:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        uid = self.users[idx]
        iid = self.items[idx]
        u_emb = self.embeds[uid]
        i_emb = self.embeds[iid]
        label = self.labels[idx]
        return uid, iid, u_emb, i_emb, label


