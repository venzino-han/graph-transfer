import torch as th
import pandas as pd
import numpy as np
import dgl

from igmc.user_item_graph import UserItemGraph


#######################
# Subgraph Extraction 
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
# Subgraph Dataset 
#######################

class UserItemDataset(th.utils.data.Dataset):
    def __init__(self, user_itme_graph: UserItemGraph, 
                hop=1, sample_ratio=1.0, max_nodes_per_hop=100):

        self.g_labels = user_itme_graph.labels
        self.graph = user_itme_graph.graph
        self.pairs = user_itme_graph.user_item_pairs

        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u_idx, i_idx = self.pairs[idx]
        subgraph = subgraph_extraction_labeling(u_idx, i_idx, self.graph)

        if 'feature' in subgraph.edata.keys():
            masked_feat = th.mul(subgraph.edata['feature'], th.unsqueeze(subgraph.edata['edge_mask'],1))
            subgraph.edata['feature']= masked_feat
        
        g_label = self.g_labels[idx]
        return subgraph, g_label


def collate_data(data):
    g_list, label_list = map(list, zip(*data))
    g = dgl.batch(g_list)
    g_label = th.stack(label_list)
    return g, g_label

NUM_WORKERS = 16

def get_dataloader(data_path, batch_size=32, feature_path=None):

    train_df = pd.read_csv(f'{data_path}_train.csv')
    valid_df = pd.read_csv(f'{data_path}_valid.csv')
    test_df = pd.read_csv(f'{data_path}_test.csv')

    #accumulate
    valid_df = pd.concat([train_df, valid_df])
    test_df = pd.concat([valid_df, test_df])

    train_graph = UserItemGraph(label_col='rating',
                                user_col='user_id',
                                item_col='item_id',
                                edge_feature_from=feature_path,
                                df=train_df,
                                edge_idx_range=(0, len(train_df)))

    train_dataset = UserItemDataset(user_itme_graph=train_graph,
                                    hop=1, sample_ratio=1.0, max_nodes_per_hop=100)

    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                            num_workers=NUM_WORKERS, collate_fn=collate_data, pin_memory=True)


    valid_graph = UserItemGraph(label_col='rating',
                            user_col='user_id',
                            item_col='item_id',
                            edge_feature_from=feature_path,
                            df=valid_df,
                            edge_idx_range=(len(train_df), len(valid_df)))

    valid_dataset = UserItemDataset(user_itme_graph=valid_graph,
                                    hop=1, sample_ratio=1.0, max_nodes_per_hop=100)

    valid_loader = th.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, 
                                            num_workers=NUM_WORKERS, collate_fn=collate_data, pin_memory=True)


    test_graph = UserItemGraph(label_col='rating',
                            user_col='user_id',
                            item_col='item_id',
                            edge_feature_from=feature_path,
                            df=test_df,
                            edge_idx_range=(len(valid_df), len(test_df)))

    test_dataset = UserItemDataset(user_itme_graph=test_graph,
                                    hop=1, sample_ratio=1.0, max_nodes_per_hop=100)

    test_loader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, 
                                            num_workers=NUM_WORKERS, collate_fn=collate_data, pin_memory=True)

    return train_loader, valid_loader, test_loader