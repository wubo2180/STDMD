import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
import learn2learn as l2l
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader,EnglandCovidDatasetLoader,PedalMeDatasetLoader,WikiMathsDatasetLoader,PemsBayDatasetLoader,WindmillOutputLargeDatasetLoader,WindmillOutputSmallDatasetLoader,MTMDatasetLoader,WindmillOutputMediumDatasetLoader,METRLADatasetLoader,MontevideoBusDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split,StaticGraphTemporalSignal
from torch_geometric.utils import negative_sampling
from  utils import *
import torch.optim as optim
from tqdm import tqdm
from model import metaDynamicGCN
from dataset import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
def compute_space_loss(embedding, index_set, criterion_space):
    # embedding = torch.relu(embedding)
    pos_score = torch.sum(embedding[index_set[0]] * embedding[index_set[1]], dim=1)
    neg_score = torch.sum(embedding[index_set[0]] * embedding[index_set[1]], dim=1)
    loss = criterion_space(pos_score, torch.ones_like(pos_score)) + \
           criterion_space(neg_score, torch.zeros_like(neg_score))
    return loss

def compute_temporal_loss(embedding, index_set, snapshot, criterion_temporal):
    
    
    y = snapshot.y[index_set]
    # embedding = torch.sigmoid(embedding[index_set]).reshape(-1)
    embedding = torch.relu(embedding[index_set]).reshape(-1)
    # print(embedding)
    # print(y)
    # print((embedding-y)**2)
    loss = criterion_temporal (embedding, y)
    # print(loss.item())
    # dd
    return loss

def train (args, model, maml, optimizer, train_dataset, criterion_space, criterion_temporal):
    

    cost = 0
    for time, snapshot in enumerate(tqdm(train_dataset,ncols=100)):

        snapshot = snapshot.to(args.device)
        # print(snapshot)
        # dd
        embedding = model(snapshot)
        task_model = maml.clone()
        query_space_loss, query_temporal_loss =0.0,0.0
        
        space_suppport_set = snapshot.pos_sup_edge_index
        space_query_set = snapshot.neg_sup_edge_index
        temporal_suppport_set = snapshot.temporal_sup_index
        temporal_query_set = snapshot.temporal_que_index
        
        for i in range(args.update_sapce_step):
            support_space_loss = compute_space_loss(embedding, space_suppport_set, criterion_space)
            # print(support_space_loss)
            task_model.adapt(support_space_loss, allow_unused=True, allow_nograd = True)
            query_space_loss += compute_space_loss(embedding, space_query_set, criterion_space)

        for i in range(args.update_temporal_step):

            suppport_temporal_loss = compute_temporal_loss(embedding,temporal_suppport_set,snapshot,criterion_temporal)
            task_model.adapt(suppport_temporal_loss, allow_unused=True, allow_nograd = True)
            query_temporal_loss += compute_temporal_loss(embedding,temporal_query_set,snapshot,criterion_temporal)

        optimizer.zero_grad()

        evaluation_loss = 0.5*query_space_loss + 0.5*query_temporal_loss
        
        evaluation_loss.backward() 
        optimizer.step()
    print('meta train loss: {:.4f}'.format(evaluation_loss.item()))

# def eval(args, model,test_dataset):
#     model.eval()
#     cost = 0
#     y_hat_list = []
#     for time, snapshot in enumerate(test_dataset):
#         snapshot = snapshot.to(args.device)
#         y_hat = model(snapshot)
#         y_hat_list.append(y_hat)
#         cost = cost + torch.mean((y_hat-snapshot.y)**2)
#     # print
#     cost = cost / (time+1)
#     cost = cost.item()
#     print("MSE: {:.4f}".format(cost))
#     loader = WikiMathsDatasetLoader()
#     dataset = loader.get_dataset()
#     y_hat_numpy = np.array(y_hat_list)
#     # Identify two connected nodes (example: nodes 0 and 1)
#     connected_nodes = (0, 1)  # Replace with actual connected nodes from dataset
#     node1, node2 = connected_nodes

#     # Extract temporal features for the connected nodes
#     time_series_node1 = [snapshot.x[node1].numpy() for snapshot in dataset][:100]
#     time_series_node2 = [snapshot.x[node2].numpy() for snapshot in dataset][:100]

#     # Assuming features are multidimensional, we visualize a specific feature (e.g., the first dimension)
#     feature_index = 0
#     actual_node1 = [t[feature_index] for t in time_series_node1]
#     actual_node2 = [t[feature_index] for t in time_series_node2]

#     # Obtain predicted values (assuming model.predict is used)
#     predicted_node1 = y_hat_numpy[:100]  # Replace with actual model prediction
#     predicted_node2 = y_hat_numpy[:100]  # Replace with actual model prediction

#     # Plot comparison of ground truth vs predicted values
#     plt.figure(figsize=(12, 6))

#     # Node 1
#     plt.subplot(1, 2, 1)
#     plt.plot(actual_node1, label=f'Actual Node {node1}', marker='o', color='blue', linestyle='dashed')
#     plt.plot(predicted_node1, label=f'Predicted Node {node1}', marker='s', color='red')
#     plt.title(f"Node {node1} - Prediction vs Actual", fontsize=16)
#     plt.xlabel("Time Steps", fontsize=14)
#     plt.ylabel("Feature Value", fontsize=14)
#     plt.legend(fontsize=12)
#     plt.grid(True)

#     # Node 2
#     plt.subplot(1, 2, 2)
#     plt.plot(actual_node2, label=f'Actual Node {node2}', marker='o', color='blue', linestyle='dashed')
#     plt.plot(predicted_node2, label=f'Predicted Node {node2}', marker='s', color='red')
#     plt.title(f"Node {node2} - Prediction vs Actual", fontsize=16)
#     plt.xlabel("Time Steps", fontsize=14)
#     plt.ylabel("Feature Value", fontsize=14)
#     plt.legend(fontsize=12)
#     plt.grid(True)

#     # Adjust layout and show plot
#     plt.tight_layout()
#     plt.show()
def eval(args, model, test_dataset):
    model.eval()
    cost = 0
    y_hat_list = []

    # Compute predictions for the entire test dataset
    with torch.no_grad():
        for time, snapshot in enumerate(test_dataset):
            snapshot = snapshot.to(args.device)
            y_hat = model(snapshot)
            y_hat_list.append(y_hat.cpu().numpy())  # Convert tensor to NumPy
            cost += torch.mean((y_hat - snapshot.y) ** 2)

    # Compute final MSE loss
    cost = cost / (time + 1)
    print("MSE: {:.4f}".format(cost.item()))

    # Convert predictions list to NumPy array
    y_hat_numpy = np.array(y_hat_list)  # Shape: (time_steps, num_nodes, num_features)

    # Select two connected nodes (adjust indices based on dataset)
    node1, node2 = 0, 1  # Modify as needed

    # Extract actual values from test dataset
    actual_node1 = [snapshot.y[node1].cpu().numpy() for snapshot in test_dataset][:100]
    actual_node2 = [snapshot.y[node2].cpu().numpy() for snapshot in test_dataset][:100]

    # # Convert to single feature array (select first feature)
    # feature_index = 0
    # actual_node1 = np.array([t[feature_index] for t in actual_node1])
    # actual_node2 = np.array([t[feature_index] for t in actual_node2])

    # Extract predicted values for the same nodes
    predicted_node1 = y_hat_numpy[:100, node1]
    predicted_node2 = y_hat_numpy[:100, node2]

    # ==== Visualization: Compare Actual vs. Predicted ====
    plt.figure(figsize=(12, 6))

    # Node 1
    plt.subplot(1, 2, 1)
    plt.plot(actual_node1, label=f'Actual Node {node1}', marker='o', linestyle='dashed', color='blue')
    plt.plot(predicted_node1, label=f'Predicted Node {node1}', marker='s', color='red')
    plt.title(f"Node {node1} - Prediction vs Actual", fontsize=20)
    plt.xlabel("Time Steps", fontsize=20)
    plt.ylabel("Feature Value", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)

    # Node 2
    plt.subplot(1, 2, 2)
    plt.plot(actual_node2, label=f'Actual Node {node2}', marker='o', linestyle='dashed', color='blue')
    plt.plot(predicted_node2, label=f'Predicted Node {node2}', marker='s', color='red')
    plt.title(f"Node {node2} - Prediction vs Actual", fontsize=20)
    plt.xlabel("Time Steps", fontsize=20)
    plt.ylabel("Feature Value", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig("figure/STDMD_vis.pdf", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # ========== Cluster Visualization ==========
    method = "TSNE"  # Choose "TSNE" or "PCA"
    
    # Extract embeddings for all time steps and nodes
    # y_hat_numpy shape: (time_steps, num_nodes, num_features)
    # We want to visualize across time for multiple nodes
    
    # Reshape to (time_steps * num_nodes, num_features) for clustering
    num_time_steps = min(100, len(y_hat_numpy))
    
    # Get all nodes' data across time
    y_hat_reshaped = y_hat_numpy[:num_time_steps].reshape(-1, y_hat_numpy.shape[-1])  # (time*nodes, features)
    actual_reshaped = np.array([snapshot.y.cpu().numpy() for snapshot in test_dataset[:num_time_steps]])
    actual_reshaped = actual_reshaped.reshape(-1, actual_reshaped.shape[-1])  # (time*nodes, features)
    
    print(f"Data shapes for clustering: y_hat={y_hat_reshaped.shape}, actual={actual_reshaped.shape}")
    
    # Check if we have enough features for dimensionality reduction
    n_features = y_hat_reshaped.shape[1]
    
    if n_features == 1:
        # If only 1 feature, create a simple 1D visualization
        print("Only 1 feature detected, creating 1D visualization...")
        plt.figure(figsize=(12, 6))
        
        # Sample some points for visualization
        sample_size = min(200, len(y_hat_reshaped))
        indices = np.random.choice(len(y_hat_reshaped), sample_size, replace=False)
        
        plt.scatter(actual_reshaped[indices], np.zeros_like(actual_reshaped[indices]), 
                   c='blue', label="Actual Values", alpha=0.6, edgecolors='k', s=100)
        plt.scatter(y_hat_reshaped[indices], np.ones_like(y_hat_reshaped[indices]) * 0.1, 
                   c='red', label="Predicted Values", alpha=0.6, edgecolors='k', s=100)
        
        plt.title("Distribution Comparison: Actual vs Predicted", fontsize=16)
        plt.xlabel("Feature Value", fontsize=14)
        plt.yticks([0, 0.1], ['Actual', 'Predicted'])
        plt.legend(fontsize=12)
        plt.grid(True, axis='x')
        
    else:
        # If we have multiple features, use t-SNE or PCA
        if method == "TSNE":
            # Adjust perplexity based on sample size
            perplexity = min(30, len(y_hat_reshaped) // 5)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        else:
            reducer = PCA(n_components=2)
        
        print(f"Applying {method} with n_components=2...")
        actual_embedded = reducer.fit_transform(actual_reshaped)
        predicted_embedded = reducer.fit_transform(y_hat_reshaped)

        # Create 2D visualization
        plt.figure(figsize=(10, 6))

        plt.scatter(actual_embedded[:, 0], actual_embedded[:, 1], 
                   c='blue', label="Actual", alpha=0.6, edgecolors='k')
        plt.scatter(predicted_embedded[:, 0], predicted_embedded[:, 1], 
                   c='red', label="Predicted", alpha=0.6, edgecolors='k')

        plt.title(f"Cluster Visualization ({method}): Actual vs. Predicted", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

    # Save the figure
    plt.savefig("figure/STDMD_cluster_vis.pdf", dpi=300, bbox_inches='tight')
    plt.show()
def main(args):
    if args.dataset == 'EnglandCovid':
        loader = EnglandCovidDatasetLoader()
    elif args.dataset == 'PedalMe':
        loader = PedalMeDatasetLoader()
    elif args.dataset == 'WikiMaths':
        loader = WikiMathsDatasetLoader()
    elif args.dataset == 'WindmillOutputLarge':
        loader = MontevideoBusDatasetLoader()

    dataset = loader.get_dataset()
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=args.train_ratio)

    train_dataset, args.num_nodes, args.input_dim = data_preprocessing(train_dataset,args)

    # print(node)
    model = metaDynamicGCN(args).to(device)
    maml = l2l.algorithms.MAML(model, lr=args.update_lr)
    optimizer = optim.Adam(maml.parameters(), lr=args.meta_lr, weight_decay=args.decay)
    criterion_space = nn.BCEWithLogitsLoss()
    criterion_temporal = nn.MSELoss(reduction='mean')
    for epoch in range(args.epochs):
        print("====epoch " + str(epoch)+'====')
        train(args, model, maml, optimizer, train_dataset, criterion_space, criterion_temporal)
        eval(args, maml,test_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='epoch number', default=50)
    parser.add_argument('--num_nodes', type=int, help='number of nodes')
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=100)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=100)
    # parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=8)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-2)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
    parser.add_argument('--update_sapce_step', type=int, help='task-level inner update steps', default=1)
    parser.add_argument('--update_temporal_step', type=int, help='update steps for finetunning', default=1)
    parser.add_argument('--decay', type=float, help='decay', default=1e-3)
    parser.add_argument('--train_ratio', type=float, help='train_ratio', default=0.8)
    parser.add_argument('--input_dim', type=int, help='input feature dim', default=4)
    parser.add_argument('--hidden_dim', type=int, help='hidden dim', default=32)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.5)   

    parser.add_argument("--num_workers", default=0, type=int, required=False, help="num of workers")

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='WikiMaths', help='dataset.')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.device = device
    print(args) 
    main(args)



# Load dataset


