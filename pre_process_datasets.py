import numpy as np
import torch
import os
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import floyd_warshall, dijkstra
import scipy
import threading


def parallel_pre_process(data, is_graph_task, data_name, processed_data_dir='processed_data'):
    print(f'Preprocessing {data_name} dataset')
    if is_graph_task:
        threads = []
        for i, g in enumerate(data):
            print(f'preprocessing graph {i}')
            g.x = torch.cat((g.x, torch.ones(g.x.size(0), 1)), dim=-1)
            g.node_distances = torch.empty((g.x.size(0), g.x.size(0)), dtype=torch.float32)
            g.normalization_matrix = torch.empty((g.x.size(0), g.x.size(0)), dtype=torch.float32)
            num_nodes = g.x.size(0)
            t = threading.Thread(target=parallel_dijkstra_and_normalization_all,
                                 args=(g.edge_index, num_nodes, g.node_distances, g.normalization_matrix))
            threads.append(t)
            t.start()

            for t in threads:
                t.join()


    else:
        if os.path.exists(f'{processed_data_dir}/{data_name}_just_distances.pt'):
            data = torch.load(f'{processed_data_dir}/{data_name}_just_distances.pt')
            normalization_matrix = data.node_distances.clone()
            node_distances = data.node_distances
        else:
            data.x = torch.cat((data.x, torch.ones(data.x.size(0), 1)), dim=-1)
            adj = scipy.sparse.lil_matrix(to_scipy_sparse_matrix(data.edge_index))
            node_distances = torch.empty((data.x.size(0), data.x.size(0)), dtype=torch.float32)
            threads = []
            for i in range(data.x.shape[0]):
                t = threading.Thread(target=parallel_dijskta_from_source, args=(adj, i, node_distances))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            node_distances = torch.nan_to_num(node_distances, posinf=np.inf)  # replace inf with 0
            node_distances += 1
            node_distances = 1 / node_distances  # replace inf with 0
            data.node_distances = node_distances

            if not os.path.exists(f'{processed_data_dir}/{data_name}_just_distances.pt'):
                torch.save(data, f'{processed_data_dir}/{data_name}_just_distances.pt')
                print(f'Saved preprocessed _just_distances {data_name} dataset')

            print('finished dijkstra parallel on all nodes')
            normalization_matrix = node_distances.clone()
        # count unique values in node_distances
        print('start normalization matrix')

        threads = []
        for i in range(data.x.shape[0]):
            print(f'normalization matrix on node {i}')
            t = threading.Thread(target=parallel_normalization_matrix, args=(normalization_matrix, node_distances, i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        data.normalization_matrix = normalization_matrix

    if not os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
        torch.save(data, f'{processed_data_dir}/{data_name}.pt')
        print(f'Saved preprocessed {data_name} dataset')


def parallel_normalization_matrix(normalization_matrix, node_distances, index):
    print(f'start normalization matrix parallel on node {index}')
    distances_counts = torch.unique(node_distances[index], return_counts=True)
    normalization_matrix[index].apply_(
        lambda x: distances_counts[1][(distances_counts[0].float() == x).nonzero().item()])
    print(f'finished normalization matrix parallel on node {index}')


def parallel_dijskta_from_source(adj, source_node, node_distances):
    node_distances[source_node] = torch.from_numpy(dijkstra(adj, indices=[source_node])).float()
    print(f'finished dijkstra parallel on node {source_node}')


def parallel_dijkstra_and_normalization_all(edge_index, num_nodes, node_distances, normalization_matrix):
    adj = scipy.sparse.lil_matrix(to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes))
    curr_node_distances = torch.from_numpy(dijkstra(adj)).float()
    curr_node_distances = torch.nan_to_num(curr_node_distances, posinf=np.inf)
    curr_node_distances += 1
    curr_node_distances = 1 / curr_node_distances
    node_distances = curr_node_distances
    normalization_matrix = node_distances.clone()
    for i, entry in enumerate(node_distances):
        distances_counts = torch.unique(entry, return_counts=True)
        for j, x in enumerate(normalization_matrix[i]):
            normalization_matrix[i][j] = distances_counts[1][(distances_counts[0] == x).nonzero().item()]
    print('parallel_dijkstra_and_normalization_all')


def pre_process(data, is_graph_task, data_name, processed_data_dir='processed_data'):
    print(f'Preprocessing {data_name} dataset')
    if is_graph_task:
        for i, g in enumerate(data):
            g.x = torch.cat((g.x, torch.ones(g.x.size(0), 1)), dim=-1)
            adj = scipy.sparse.lil_matrix(to_scipy_sparse_matrix(g.edge_index, num_nodes=g.x.size(0)))
            node_distances = torch.from_numpy(dijkstra(adj)).float()

            node_distances = torch.nan_to_num(node_distances, posinf=np.inf)
            node_distances += 1
            node_distances = 1 / node_distances
            g.node_distances = node_distances

            normalization_matrix = node_distances.clone()
            for i, entry in enumerate(node_distances):
                distances_counts = torch.unique(entry, return_counts=True)
                normalization_matrix[i].apply_(
                    lambda x: distances_counts[1][(distances_counts[0] == x).nonzero().item()])
            g.normalization_matrix = normalization_matrix



    else:
        data.x = torch.cat((data.x, torch.ones(data.x.size(0), 1)), dim=-1)
        adj = scipy.sparse.lil_matrix(to_scipy_sparse_matrix(data.edge_index))
        node_distances = torch.from_numpy(dijkstra(adj)).float()
        node_distances = torch.nan_to_num(node_distances, posinf=np.inf)
        node_distances += 1
        node_distances = 1 / node_distances
        data.node_distances = node_distances

        # count unique values in node_distances
        normalization_matrix = node_distances.clone()
        for i, entry in enumerate(node_distances):
            distances_counts = torch.unique(entry, return_counts=True)
            normalization_matrix[i].apply_(
                lambda x: distances_counts[1][(distances_counts[0].float() == x).nonzero().item()])

        data.normalization_matrix = normalization_matrix

    if not os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
        torch.save(data, f'{processed_data_dir}/{data_name}.pt')
        print(f'Saved preprocessed {data_name} dataset')
