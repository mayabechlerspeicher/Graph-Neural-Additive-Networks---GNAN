import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset, QM9  # , LRGBDataset
from ogb.nodeproppred import PygNodePropPredDataset
import csv
from torch_geometric.data import Data
import torch
import numpy as np
import torch_geometric as pyg
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from pre_process_datasets import *

def load_pubmed():
    # Add labels and features
    file_path = 'data/Pubmed-Diabetes/Pubmed-Diabetes.NODE.paper.tab'

    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader, None)
        row = next(reader)
        all_features = [f.split(":")[1] for f in row if 'numeric' in f]

    pubmed_data = dict()

    # Add labels and features
    file_path = 'data/Pubmed-Diabetes/Pubmed-Diabetes.NODE.paper.tab'
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader, None)
        next(reader, None)
        for i, row in enumerate(reader):
            document_id = row[0]
            document_label = int(row[1].split("=")[1])
            features = {feature.split('=')[0]: round(float(feature.split('=')[1]), 3) for feature in row[2:-1]}

            pubmed_data[document_id] = dict()
            pubmed_data[document_id]['label'] = document_label - 1
            pubmed_data[document_id]['features'] = features
            pubmed_data[document_id]['edge_to'] = []

    nodes = sorted(pubmed_data.keys())
    all_features = sorted(all_features)

    # Add Edges
    file_path = 'data/Pubmed-Diabetes/Pubmed-Diabetes.DIRECTED.cites.tab'

    sources = []
    sinks = []

    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader, None)
        next(reader, None)
        for row in reader:
            source = row[1].split(':')[1]
            sink = row[3].split(':')[1]
            sources.append(nodes.index(source))
            sinks.append(nodes.index(sink))

    # Construct edge index
    sources = torch.tensor(sources)
    sinks = torch.tensor(sinks)
    edge_index = torch.stack([sources, sinks], dim=0)

    # Construct feature matrix
    x = torch.zeros((len(nodes), len(all_features)))

    for node_indx, node_name in enumerate(nodes):
        node_features = pubmed_data[node_name]['features']
        for feature, val in node_features.items():
            feature_inx = all_features.index(feature)
            x[node_indx, feature_inx] = val

    # Construct label vector
    y = torch.zeros(len(nodes))
    for node_indx, node_name in enumerate(nodes):
        y[node_indx] = pubmed_data[node_name]['label']

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def get_data(data_name, processed_data_dir='processed_data', seed_index=0):
    print(f'Loading {data_name} dataset')
    data_path = 'data'
    compute_auc = False
    train_loader, val_loader, test_loader = None, None, None
    num_features = None

    if data_name == 'cora':
        if os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
            data = torch.load(f'{processed_data_dir}/{data_name}.pt')
        else:
            dataset = Planetoid(data_path, 'Cora', transform=T.NormalizeFeatures())
            data = dataset[0]
            pre_process(data, False, data_name, processed_data_dir=processed_data_dir)
        is_graph_task = False
        is_regression = False
        num_classes = 7

    elif data_name == 'citeseer':
        if os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
            data = torch.load(f'{processed_data_dir}/{data_name}.pt')
        else:
            dataset = Planetoid(data_path, 'Citeseer', transform=T.NormalizeFeatures())
            data = dataset[0]
            pre_process(data, False, data_name, processed_data_dir=processed_data_dir)
        is_graph_task = False
        is_regression = False
        num_classes = 6

    elif data_name == 'pubmed':
        if os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
            data = torch.load(f'{processed_data_dir}/{data_name}.pt')
        else:
            dataset = load_pubmed()
            data = dataset[0]
            pre_process(data, False, data_name, processed_data_dir=processed_data_dir)
        is_graph_task = False
        is_regression = False
        num_classes = 3

    elif data_name == 'mutagenicity':
        if os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
            data = torch.load(f'{processed_data_dir}/{data_name}.pt')
        else:
            dataset = TUDataset(root=data_path, name='Mutagenicity')
            data = list(dataset)
            pre_process(data, True, data_name, processed_data_dir=processed_data_dir)
        is_graph_task = True
        is_regression = False
        num_classes = 2

    elif data_name == 'proteins':
        if os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
            data = torch.load(f'{processed_data_dir}/{data_name}.pt')
        else:
            dataset = TUDataset(root=data_path, name='PROTEINS')
            data = list(dataset)
            pre_process(data, True, data_name, processed_data_dir=processed_data_dir)
        is_graph_task = True
        is_regression = False
        num_classes = 2

    elif data_name == 'ptc_mr':
        if os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
            data = torch.load(f'{processed_data_dir}/{data_name}.pt')
        else:
            dataset = TUDataset(root=data_path, name='PTC_MR')
            data = list(dataset)
            pre_process(data, True, data_name, processed_data_dir=processed_data_dir)
        is_graph_task = True
        is_regression = False
        num_classes = 2

    elif data_name == 'QM9_data_3':
        if os.path.exists(f'{processed_data_dir}/{data_name}_batches.pt'):
            data = torch.load(f'{processed_data_dir}/{data_name}_batches.pt')
            train_loader = torch.load(f'{processed_data_dir}/{data_name}_train_batches.pt')
            val_loader = torch.load(f'{processed_data_dir}/{data_name}_val_batches.pt')
            test_loader = torch.load(f'{processed_data_dir}/{data_name}_test_batches.pt')
            num_features = train_loader.dataset[0].x.size(-1)
        else:
            dataset = QM9(root='data/QM9')
            data = list(dataset)
            parallel_pre_process(data, True, data_name, processed_data_dir=processed_data_dir)

        for g in train_loader.dataset:
            g.y = g.y[:, 2].unsqueeze(1)
        for g in val_loader.dataset:
            g.y = g.y[:, 2].unsqueeze(1)
        for g in test_loader.dataset:
            g.y = g.y[:, 2].unsqueeze(1)

        skf = sklearn.model_selection.KFold(n_splits=5)
        splits_list = list(skf.split(data))
        train_idx, test_idx = splits_list[seed_index][0], splits_list[seed_index][1]
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]
        # print precents of each class in the train and test data
        train_labels = np.array([g.y.numpy() for g in train_data]).flatten()
        test_labels = np.array([g.y.numpy() for g in test_data]).flatten()
        print(f'train labels: {np.unique(train_labels, return_counts=True)}')
        print(f'test labels: {np.unique(test_labels, return_counts=True)}')

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
        val_loader = pyg.loader.DataLoader(test_data, batch_size=32)
        test_loader = pyg.loader.DataLoader(test_data, batch_size=32)

        is_graph_task = True
        is_regression = True
        num_classes = 1

    elif data_name == 'QM9_data_2':
        if os.path.exists(f'{processed_data_dir}/{data_name}_batches.pt'):
            data = torch.load(f'{processed_data_dir}/{data_name}_batches.pt')
            train_loader = torch.load(f'{processed_data_dir}/{data_name}_train_batches.pt')
            val_loader = torch.load(f'{processed_data_dir}/{data_name}_val_batches.pt')
            test_loader = torch.load(f'{processed_data_dir}/{data_name}_test_batches.pt')
            num_features = train_loader.dataset[0].x.size(-1)
        else:
            dataset = QM9(root='data/QM9')
            data = list(dataset)
            parallel_pre_process(data, True, data_name, processed_data_dir=processed_data_dir)

        for g in train_loader.dataset:
            g.y = g.y[:, 1].unsqueeze(1)
        for g in val_loader.dataset:
            g.y = g.y[:, 1].unsqueeze(1)
        for g in test_loader.dataset:
            g.y = g.y[:, 1].unsqueeze(1)


        skf = sklearn.model_selection.KFold(n_splits=5)
        splits_list = list(skf.split(data))
        train_idx, test_idx = splits_list[seed_index][0], splits_list[seed_index][1]
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]
        # print precents of each class in the train and test data
        train_labels = np.array([g.y.numpy() for g in train_data]).flatten()
        test_labels = np.array([g.y.numpy() for g in test_data]).flatten()
        print(f'train labels: {np.unique(train_labels, return_counts=True)}')
        print(f'test labels: {np.unique(test_labels, return_counts=True)}')

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
        val_loader = pyg.loader.DataLoader(test_data, batch_size=32)
        test_loader = pyg.loader.DataLoader(test_data, batch_size=32)

        is_graph_task = True
        is_regression = True
        num_classes = 1

    elif data_name == 'QM9_data_1':
        if os.path.exists(f'{processed_data_dir}/{data_name}_train_batches.pt'):
            train_loader = torch.load(f'{processed_data_dir}/{data_name}_train_batches.pt')
            val_loader = torch.load(f'{processed_data_dir}/{data_name}_val_batches.pt')
            test_loader = torch.load(f'{processed_data_dir}/{data_name}_test_batches.pt')
            num_features = train_loader.dataset[0].x.size(-1)

            for g in train_loader.dataset:
                g.y = g.y[:, 0].unsqueeze(1)
            for g in val_loader.dataset:
                g.y = g.y[:, 0].unsqueeze(1)
            for g in test_loader.dataset:
                g.y = g.y[:, 0].unsqueeze(1)
        else:
            dataset = QM9(root='data/QM9')
            data = list(dataset)
            parallel_pre_process(data, True, data_name, processed_data_dir=processed_data_dir)

        for g in data:
            g.y = g.y[:, 1].unsqueeze(1)

        skf = sklearn.model_selection.KFold(n_splits=5)
        splits_list = list(skf.split(data))
        train_idx, test_idx = splits_list[seed_index][0], splits_list[seed_index][1]
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]
        # print precents of each class in the train and test data
        train_labels = np.array([g.y.numpy() for g in train_data]).flatten()
        test_labels = np.array([g.y.numpy() for g in test_data]).flatten()
        print(f'train labels: {np.unique(train_labels, return_counts=True)}')
        print(f'test labels: {np.unique(test_labels, return_counts=True)}')

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
        val_loader = pyg.loader.DataLoader(test_data, batch_size=32)
        test_loader = pyg.loader.DataLoader(test_data, batch_size=32)

        is_graph_task = True
        is_regression = True
        num_classes = 1

    elif data_name == 'ogb_arxiv':
        if os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
            data = torch.load(f'{processed_data_dir}/{data_name}.pt')
        else:

            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data')
            split_idx = dataset.get_idx_split()
            data = dataset[0]
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[split_idx['train']] = True
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask[split_idx['valid']] = True
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask[split_idx['test']] = True
            parallel_pre_process(data, False, data_name)

        is_graph_task = False
        is_regression = True
        num_classes = 1

    elif data_name == 'tolokers':
        if os.path.exists(f'{processed_data_dir}/{data_name}.pt'):
            data = torch.load(f'{processed_data_dir}/{data_name}.pt')
            data.train_mask = data.train_mask[:, 0].T
            data.val_mask = data.val_mask[:, 0].T
            data.test_mask = data.test_mask[:, 0].T
        else:
            data = torch.load('data/tolokers/processed/data.pt')[0]
            data = Data(x=data['x'], y=data['y'], edge_index=data['edge_index'], train_mask=data['train_mask'],
                        val_mask=data['val_mask'], test_mask=data['test_mask'])

            pre_process(data, False, data_name, processed_data_dir=processed_data_dir)
        num_classes = 2
        is_regression = False
        is_graph_task = False
        compute_auc = True

    else:
        raise ValueError('Unknown dataset')
    if train_loader is None:
        if is_graph_task:
            labels = np.array([g.y.numpy() for g in data]).flatten()

            if is_regression:
                skf = sklearn.model_selection.KFold(n_splits=5)
                splits_list = list(skf.split(data))
            else:
                skf = StratifiedKFold(n_splits=5)
                splits_list = list(skf.split(data, labels))
            train_idx, test_idx = splits_list[seed_index][0], splits_list[seed_index][1]
            train_data = [data[i] for i in train_idx]
            test_data = [data[i] for i in test_idx]

            # print precents of each class in the train and test data
            train_labels = np.array([g.y.numpy() for g in train_data]).flatten()
            test_labels = np.array([g.y.numpy() for g in test_data]).flatten()
            print(f'train labels: {np.unique(train_labels, return_counts=True)}')
            print(f'test labels: {np.unique(test_labels, return_counts=True)}')

            # split the train data to train and val
            stratify = None if is_regression else train_labels
            train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2,
                                                                              random_state=42, stratify=stratify)
            print(f'train labels: {np.unique(train_labels, return_counts=True)}')
            print(f'val labels: {np.unique(val_labels, return_counts=True)}')

            train_loader = pyg.loader.DataLoader(train_data, batch_size=1)
            val_loader = pyg.loader.DataLoader(val_data, batch_size=1)
            test_loader = pyg.loader.DataLoader(test_data, batch_size=1)
        else:
            # else:
            train_data = data
            val_data = data
            test_data = data
            train_loader = pyg.loader.DataLoader([train_data], batch_size=1)
            val_loader = pyg.loader.DataLoader([val_data], batch_size=1)
            test_loader = pyg.loader.DataLoader([test_data], batch_size=1)

    if num_features is None:
        num_features = train_data[0].x.size(-1) if is_graph_task else train_data.x.size(-1)

    return train_loader, val_loader, test_loader, num_features, num_classes, is_graph_task, is_regression, compute_auc
