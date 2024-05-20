import torch_geometric as pyg
import numpy as np
import torch
import random

def get_moments(Gs):
    std_mean = 0
    total_mean = 0
    skew_mean = 0
    kurt_mean = 0
    for g in Gs:

        degrees = np.array([d for n, d in g.degree()])
        std_mean += degrees.std()
        total_mean += degrees.mean()
        skew_mean += skew(degrees)
        kurt_mean += kurtosis(degrees)
    return (total_mean / len(Gs)), (std_mean / len(Gs)), (skew_mean / len(Gs)), (kurt_mean / len(Gs))


def reduce_ratio_add_random_edges_to_pyg_graphs(pyg_graphs, reduce_per=None, reduce_goal=None):
    nx_graphs = [pyg.utils.to_networkx(graph, to_undirected=True) for graph in pyg_graphs]
    # new_nx_graphs = []
    new_pyg_graphs = []
    for i, pyg_graph in enumerate(pyg_graphs):
        graph = pyg_graph.clone()
        nx_graph = nx_graphs[i]
        new_mean, new_std, new_skew, new_kurt = get_moments([nx_graph])
        curr_ratio = new_std / new_mean
        if reduce_per is not None:
            goal_ratio = curr_ratio - reduce_per * curr_ratio
        else:
            goal_ratio = reduce_goal
        while curr_ratio > goal_ratio:
            degrees_sorted_tuples = sorted(list(nx_graph.degree), key=lambda x: x[1])
            low_deg_nodes = [x[0] for x in degrees_sorted_tuples[:10]]
            possible_edges = list(nx.non_edges(nx_graph))
            possible_edges_low_deg_nodes = []
            for edge in possible_edges:
                if edge[0] in low_deg_nodes or edge[1] in low_deg_nodes:
                    possible_edges_low_deg_nodes.append(edge)
            if len(possible_edges_low_deg_nodes) == 0:
                new_mean, new_std, new_skew, new_kurt = get_moments([nx_graph])
                print('num of nodes in graph:' + str(len(nx_graph.nodes)))
                print('new mean:' + str(new_mean))
                print('new std:' + str(new_std))
                print('new skew:' + str(new_skew))
                print('new kurt:' + str(new_kurt))
                print('no possible edges')
                print('num of edges:' + str(len(nx_graph.edges)))
                print(nx_graph.edges)

            random_edge = random.choice(possible_edges_low_deg_nodes)
            graph.edge_index = torch.cat([graph.edge_index, torch.tensor([[random_edge[0], random_edge[1]], [random_edge[1], random_edge[0]]]).T], dim=1)
            graph.edge_weight = torch.cat([graph.edge_weight, torch.tensor([0.5, 0.5])])
            nx_graph.add_edge(*random_edge)
            new_mean, new_std, new_skew, new_kurt = get_moments([nx_graph])
            curr_ratio = new_std / new_mean
        # new_nx_graphs.append(nx_graph)
        new_pyg_graphs.append(graph)
    return new_pyg_graphs, nx_graphs



# # if __name__ == "__main__":
#
#     self.dataset = GraphDataset(torch.load(
#         self.processed_dir / f"{self.name}.pt"))
#
#     goal_train_ratio = train_std_mean_ratio - percentage_to_reduce * train_std_mean_ratio
#     if goal_train_ratio < 0.05:
#         break
#     train_pyg_graphs, train_nx_graphs = exp_utils.reduce_ratio_add_random_edges_to_pyg_graphs(train_pyg_graphs,
#                                                                                               reduce_goal=goal_train_ratio)
#
#     train_mean, train_std, train_skew, train_kurt = get_moments(train_nx_graphs)
#     train_std_mean_ratio = train_std / train_mean
#     print('new train ratio: ', train_std_mean_ratio)
#
#     test_pyg_graphs, test_nx_graphs = reduce_ratio_add_random_edges_to_pyg_graphs(test_pyg_graphs,
#                                                                                             reduce_goal=goal_train_ratio)
#
#     test_mean, test_std, test_skew, test_kurt = get_moments(test_nx_graphs)
#     test_std_mean_ratio = test_std / test_mean
#     print('new test ratio: ', test_std_mean_ratio)
