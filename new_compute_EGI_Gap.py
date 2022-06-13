import argparse, time
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import scipy.sparse as sp
from scipy.linalg import eigh, norm
from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm
import pickle
from collections import defaultdict
from random import shuffle
from csv import reader


# def read_struct_net(file_path):
def get_nx_graph(file_path):
    # get empty nx graph
    g = nx.Graph()

    # read data from csv
    df = pd.read_csv(file_path, index_col=0).iloc[:, :2]
    # remove duplicate and reset the item_id
    df = df.drop_duplicates()
    df['item_id'] += max(df['user_id']) + 1

    # add edge to nx graph
    for user_id, item_id in zip(df['user_id'], df['item_id']):
        g.add_edge(int(user_id), int(item_id))

    print('num of nodes : ', g.number_of_nodes())
    print('num of edges : ', g.number_of_edges())

    return g
    

# def constructDGL(graph):
def get_DGL(graph):

    # get node_dictionary if there pass the num
    """
    ex) there are passed number 4, 5, 7
    0, 1, 2, 3, 6, 8, 9
    -> 
    0:0, 1:1, 2:2, 3:3, 6:4, 8:5, 9:6
    """
    node_mapping = defaultdict(int)
    for node in sorted(list(graph.nodes())):
        node_mapping[node] = len(node_mapping)
    """
    node_mapping = defaultdict(int,
                    {0: 0,
                    1: 1,
                    2: 2,
                    3: 3,
                    4: 4,
                    ...
    """

    # create Graph using DGL Library
    new_g = DGLGraph()
    new_g.add_nodes(len(node_mapping))

    # add edge to DGL graph
    for edge in graph.edges():
        if not new_g.has_edge_between(node_mapping[edge[0]], node_mapping[edge[1]]):
            new_g.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
        if not new_g.has_edge_between(node_mapping[edge[1]], node_mapping[edge[0]]):
            new_g.add_edge(node_mapping[edge[1]], node_mapping[edge[0]])
    
    return new_g 


def compute_term(l, r):
    n = l.shape[0]

    eval_ = eigh((l-r).T @ (l-r), eigvals_only=True)
    return np.sqrt(max(eval_))
    
def main(args):
    def constructSubG(file_path):
        ## node, edge update using networkX library
        # g = read_struct_net(file_path)
        g = get_nx_graph(file_path)
        
        ## remove selfloop edges
        g.remove_edges_from(nx.selfloop_edges(g))
        print('removed selfloop edges (count) :', g.number_of_edges())
        print('----------------------')
        
        # g = constructDGL(g)
        g = get_DGL(g)
        print('dgl graph :', g)
        print('----------------------')
        g.readonly()

        node_sampler = dgl.contrib.sampling.NeighborSampler(g, 1, 10,  # 0,
                                                                neighbor_type='in', num_workers=1,
                                                                add_self_loop=False,
                                                                num_hops=args.n_layers + 1,
                                                                # shuffle=True
                                                                )
        return g, node_sampler


    def constructHopdic(ego_g):
        hop_dic = dict()

        node_set = set([])
        # layers default: 1
        """
        n 번째 홉부터 역순으로 센터까지 존재하는 노드 리스트를 dictionary 형태로 넣음
        이 후 node_set 에도 set형태로 넣어주며 n-1 ~ 0 hop 까지 중복되는 node 는 제거해줌  

        ex)
        2번째 홉부터 존재하는 노드 리스트를 넣음
        1번째 dict에 2번째 홉에 존재하는 노드는 제거하고 넣음
        즉 2번째 홉과 1번째 홉에 각각 같은 노드가 있을 경우 2번째 홉 dict 에만 포함
        """

        for layer_id in range(args.n_layers+2)[::-1]:
            hop_dic[layer_id] = list(set(ego_g.layer_parent_nid(layer_id).numpy())
                                    - node_set)
            # union
            node_set |= set(hop_dic[layer_id])

        return hop_dic

    def constructIdxCoding(hop_dic):
        idx_coding = {"00":[]}

        for layer_id in range(args.n_layers+2)[::-1]:
            for node_id in hop_dic[layer_id]:
                if node_id != "00":
                    idx_coding[node_id] = len(idx_coding) + len(idx_coding["00"]) - 1
                    # degree.append(g.in_degree())
                else:
                    idx_coding["00"] += [len(idx_coding) + len(idx_coding["00"]) - 1]
        return idx_coding
    
    def constructL(g, ego_g, idx_coding, neighbor_type='out'):
        dim = len(idx_coding) + len(idx_coding["00"]) - 1
        A = np.zeros([dim, dim])

        for i in range(ego_g.num_blocks):
            u,v = g.find_edges(ego_g.block_parent_eid(i))
            for left_id, right_id in zip(u.numpy().tolist(), v.numpy().tolist()):
                A[idx_coding[left_id], idx_coding[right_id]] = 1
    
        # lower part is the out-degree direction
        if neighbor_type=='in':
            # upper     
            A = A.T

        # select the non-zero submatrix
        selector = list(set(np.arange(dim)) - set(idx_coding['00']))
        ## select 는 ['00']을 뺀 것들의 idx, 그것을 이용해 sub-matrix 생성 
        ## 이것이 라플라시안 행렬의 Adjacency matrix 로 쓰임
        A_full = A[np.ix_(selector, selector)]

        # find L
        ## 행으로 더해서 다시 단위행렬 위치로 옮김
        D = np.diag(A_full.sum(1))
        ## 라플라시안 matrix 구하는 식
        L = D - A_full
        ## 단위행렬을 인접행렬 행합의 제곱근으로 나눔
        D_ = np.diag(1.0 / np.sqrt(A_full.sum(1)))
        ## 이때 inf 로 가는 것들은 (기존에 인접한 것이 없는 경우) 0으로 변환
        D_ = np.nan_to_num(D_, posinf=0, neginf=0) #set inf to 0
        ## 위에서 구한 L(라플라시안)을 정규화한다고 생각하면 됨
        normailized_L = np.matmul(np.matmul(D_, L), D_)
        # reassign the calculated Laplacian
        A[np.ix_(selector, selector)] = normailized_L

        if np.isnan(A.sum()):
            embed()

        return A

    def degPermute(ego_g, hop_dic, layer_id):
        if layer_id == 0:
            return hop_dic[layer_id]
        else:
            s, arg_degree_sort = torch.sort(-ego_g.layer_in_degree(layer_id))

            return torch.tensor(hop_dic[layer_id])[arg_degree_sort].tolist()

    def pad_nbhd(lg, rg, lego_g, rego_g, perm_type='shuffle', neighbor_type='out'):
        # returns two padded Laplacian
        ## hop dictionary 구함
        lhop_dic = constructHopdic(lego_g)
        rhop_dic = constructHopdic(rego_g)

        # make even the size of nhbd
        """
        hop 당 포함된 node 갯수 차이  l - r 를 구하고
        차이나는 갯수 만큼 [00] 으로 패딩
        """
        for layer_id in range(args.n_layers+2)[::-1]:
            diff = len(lhop_dic[layer_id]) - len(rhop_dic[layer_id])

            ## [00] 을 넣고 shuffle
            if perm_type == 'shuffle': # including the padded terms
                if diff>0: # l > r
                    rhop_dic[layer_id] += ["00"] * abs(diff)
                elif diff<0: # l < r
                    lhop_dic[layer_id] += ["00"] * abs(diff)
                
                
                shuffle(lhop_dic[layer_id])
                shuffle(rhop_dic[layer_id])

            ## torch.sort 로 정렬 후 [00] 을 뒤에 추가
            elif perm_type == 'degree':
                lhop_dic[layer_id] = degPermute(lego_g, lhop_dic, layer_id)
                rhop_dic[layer_id] = degPermute(rego_g, rhop_dic, layer_id)

                if diff>0:
                    rhop_dic[layer_id] += ["00"] * abs(diff)
                elif diff<0:
                    lhop_dic[layer_id] += ["00"] * abs(diff)

            ## 그냥 [00] 을 뒤에 추가
            else:
                if diff>0:
                    rhop_dic[layer_id] += ["00"] * abs(diff)
                elif diff<0:
                    lhop_dic[layer_id] += ["00"] * abs(diff)


        # construct coding dict
        ## 그래프의 모든 노드 인덱싱 n-hop 에 있는 노드부터
        """
        lidx_coding : {
                        '00': [6, 25, 40, 44, 47, 64], 305: 0, 299: 1, 33: 2, 20: 3, 
                        61: 4, 117: 5, 17: 7, 74: 8, 65: 9, 118: 10, 79: 11, 266: 12, 157: 13, 208: 14, 
                        122: 15, 167: 16, 159: 17, 66: 18, 11: 19, 111: 20, 46: 21, 37: 22
                        }
        """
        lidx_coding = constructIdxCoding(lhop_dic)
        ridx_coding = constructIdxCoding(rhop_dic)

        ## normalized Laplacian matrix return
        lL = constructL(lg, lego_g, lidx_coding, neighbor_type=neighbor_type)
        rL = constructL(rg, rego_g, ridx_coding, neighbor_type=neighbor_type)

        return lL, rL


    print('\n start!!!')
    print('---------------------')
    print('dataset path_1, dataset path_2')
    print(args.data_path_1, args.data_path_2)


    print('---------------------')
    print('dgl version : ', dgl.__version__)
    print('---------------------')

    Lg, Lego_list = constructSubG(args.data_path_1)
    Rg, Rego_list = constructSubG(args.data_path_2)


    print('Lg')
    print(Lg)

    print('----------------------')

    print('Lego_list')
    print(Lego_list)
    print('----------------------')

    # embed()
    bound = 0
    cntl = 0
    cntr = 0
    for lego_g in tqdm(Lego_list):

        print('----------------------')
        lego_block_pa_1 = lego_g.block_parent_eid(1)
        print('lego_g.block_parent_eid(1) :', lego_block_pa_1)

        print('----------------------')
        print('g.find_edges(lego_g.block_parent_eid(1)) :', Lg.find_edges(lego_block_pa_1))

        u,v = Lg.find_edges(lego_g.block_parent_eid(1))

        print('u', u)
        print('v', v)


        # tmp
        # print('break')
        # break

        cntl += 1
        cntr = 0
        for rego_g in Rego_list:
            cntr += 1
            lL, rL = pad_nbhd(Lg, Rg, lego_g, rego_g,
                                perm_type='shuffle',
                                neighbor_type='in')           

            bound += compute_term(lL, rL)

    result = bound / (cntl * cntr) 
    print(result)

    with open(f"{args.result_path}", "w") as file:
        file.write(f"result : {result}")



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
    parser.add_argument("--data-id", type=str,
                    help="[usa, europe, brazil]")
    parser.add_argument("--data-src", type=str, default='',
                    help="[usa, europe, brazil]")
    parser.add_argument("--data-path-1", type=str,
                        help="graph_1 path")
    parser.add_argument("--data-path-2", type=str,
                        help="graph_2 path")
    parser.add_argument("--result-path", type=str,
                        help="result path")                        
    parser.add_argument("--model-id", type=int, default=0,
                    help="[0, 1, 2, 3]")

    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    
    
    main(args)
