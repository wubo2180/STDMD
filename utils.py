from torch_geometric.utils import negative_sampling,structured_negative_sampling
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import torch
import numpy as np
import random
def data_preprocessing(dataset,args):
    data_list = [] 
    for time, snapshot in enumerate(dataset):
        # print(negative_sampling(snapshot.edge_index))
        data_list.append(task_construct(snapshot,args))
    # print(len(data_list))
    return data_list,snapshot.x.shape[0],snapshot.x.shape[1]


def task_construct(data,args):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    
    # sample support set and query set for each data/task/graph
    # num_sampled_edges = args.n_way * (args.k_spt + args.k_qry)
    num_sampled_edges = args.k_spt + args.k_qry
    
    # 限制采样边数不超过实际边数
    num_sampled_edges = min(num_sampled_edges, num_edges)
    
    perm = np.random.randint(num_edges, size=num_sampled_edges)
    pos_edges = data.edge_index[:, perm]

    # 尝试负采样
    neg_edges = negative_sampling(data.edge_index, num_nodes, num_sampled_edges)
    cur_num_neg = neg_edges.shape[1]
    
    # 处理负采样结果
    if cur_num_neg == 0:
        # 图太密集，无法生成真正的负边
        # 策略：使用正边的"弱版本"作为负样本
        # 方法1: 随机打乱边的端点
        print(f"⚠️ 警告: 图太密集无法生成负边，使用打乱的边作为负样本")
        neg_edges = pos_edges.clone()
        # 随机打乱目标节点
        shuffled_indices = torch.randperm(neg_edges.shape[1])
        neg_edges[1, :] = neg_edges[1, shuffled_indices]
        
    elif cur_num_neg < num_sampled_edges:
        print(f"⚠️ 警告: 负采样数量不足 ({cur_num_neg} < {num_sampled_edges})，使用有放回采样")
        # 使用有放回的随机采样
        perm = np.random.choice(cur_num_neg, size=num_sampled_edges, replace=True)
        neg_edges = neg_edges[:, perm]
    elif cur_num_neg != num_sampled_edges:
        # 负采样数量充足，随机选择
        perm = np.random.randint(cur_num_neg, size=num_sampled_edges)
        neg_edges = neg_edges[:, perm]
    
    # 确保 k_spt 不超过采样边数
    actual_k_spt = min(args.k_spt, num_sampled_edges // 2)
    actual_k_qry = num_sampled_edges - actual_k_spt

    data.pos_sup_edge_index = pos_edges[:, :actual_k_spt]
    data.neg_sup_edge_index = neg_edges[:, :actual_k_spt]
    data.pos_que_edge_index = pos_edges[:, actual_k_spt:]
    data.neg_que_edge_index = neg_edges[:, actual_k_spt:]
    
    num_sampled_nodes = min(args.k_spt + args.k_qry, num_nodes)
    perm = np.random.randint(num_nodes, size=num_sampled_nodes)
    actual_node_k_spt = min(args.k_spt, num_sampled_nodes // 2)
    data.temporal_sup_index = perm[:actual_node_k_spt]
    data.temporal_que_index = perm[actual_node_k_spt:]
    # data.aug_feature = aug_random_mask(data.x)
    return data

def aug_random_mask(input_feature, drop_percent=0.2,dim_drop=0.5):
    
    node_num, dim = input_feature.shape
    # print(dim)

    mask_num = int(node_num * drop_percent)
    mask_dim = int(dim * dim_drop)
    
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    dim_idx = [i for i in range(dim)]
    # print(mask_idx,dim_idx,mask_dim)
    # aug_feature = copy.deepcopy(input_feature)
    aug_feature = input_feature
    zeros = torch.zeros_like(aug_feature[0][0])
    for i in mask_idx:
        # for j in mask_dim:
        # mask_dim = random.sample(dim_idx, mask_dim)
        # print(mask_dim)
        aug_feature[i][random.sample(dim_idx, mask_dim)] = 0
    return aug_feature
