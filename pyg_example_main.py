from pre_process_datasets import ShortestPathDistanceTRansform
from torch_geometric.datasets import TUDataset
import torch
from GNAN import GNAN, TensorGNAN
import trainer
import torch_geometric as pyg

data_path = 'data'
dataset = TUDataset(root=data_path, name='Mutagenicity', transform=ShortestPathDistanceTRansform())
is_graph_task = True
is_regression = False
num_classes = 2
num_features = dataset.num_features
hidden_channels = 64
n_layers = 3
device = 'cpu'
dropout = 0.3
out_dim = 1
num_epochs = 10
lr = 0.0001
wd = 0.00005
loss_type = torch.nn.BCEWithLogitsLoss
model = TensorGNAN(in_channels=num_features,
             hidden_channels=hidden_channels, n_layers=n_layers,
             out_channels=out_dim, dropout=dropout, device=device, is_graph_task=True, normalize_rho=False)

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
loss = loss_type()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.9,
    patience=100,
    verbose=True,
    min_lr=1e-8,
)
train_loader =  pyg.loader.DataLoader(dataset[:100], batch_size=1)
val_loader = pyg.loader.DataLoader(dataset[:100], batch_size=1)
test_loader = pyg.loader.DataLoader(dataset[:100], batch_size=1)
for epoch in range(num_epochs):
    train_loss, train_acc, train_auc = trainer.train_epoch(model, dloader=train_loader,
                                                           loss_fn=loss,
                                                           optimizer=optimizer,
                                                           classify=~is_regression, device=device,
                                                           compute_auc=False, is_graph_task=is_graph_task)
    val_loss, val_acc, val_auc = trainer.test_epoch(model, dloader=val_loader,
                                                    loss_fn=loss, classify=~is_regression,
                                                    device=device, val_mask=True, compute_auc=False,
                                                    is_graph_task=is_graph_task)

    print(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
    scheduler.step(train_loss)
