import argparse
from models import *
import wandb
import trainer
import datasets
import uuid
import numpy as np
import torch

np.random.seed(0)
seeds = np.random.randint(low=0, high=10000, size=5)

class EarlyStopping:
    def __init__(self, metric_name, patience=3, min_is_better=False):
        self.metric_name = metric_name
        self.patience = patience
        self.min_is_better = min_is_better
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def reset(self):
        self.counter = 0

    def __call__(self, score):
        if self.min_is_better:
            score = -score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def run_exp(train_loader, val_loader, test_loader, num_features, seeds, n_layers, early_stop_flag, dropout, model_name,
            num_epochs, wandb_flag, wd,
            hidden_channels, lr, bias, patience, loss_thresh, debug, data_name, unique_run_id, one_m, normalize_m,
            is_graph_task, num_classes, final_agg, out_dim, readout_n_layers=0, is_for_plot=False,
            is_regression=False, processed_data_dir='processed_data', compute_auc=False):
    if debug:
        wandb_flag = False

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    seeds = seeds[args.seed:args.seed + 1]
    for i, seed in enumerate(seeds):

        if model_name == 'sage':
            model = GraphSAGEModel(in_channels=num_features,
                             hidden_channels=args.hidden_channels, num_layers=args.n_layers,
                             out_channels=out_dim)
        elif model_name == 'gin':
            model = GINModel(in_channels=num_features,
                             hidden_channels=hidden_channels, num_layers=n_layers,
                             out_channels=out_dim, dropout=dropout, bias=bias)

        elif model_name == 'gatv2':
            model = GATv2Model(in_channels=num_features,
                               hidden_channels=hidden_channels, num_layers=n_layers,
                               out_channels=out_dim, dropout=dropout, bias=bias)

        elif model_name == 'graphconv':
            model = GraphConvModel(in_channels=num_features,
                                   hidden_channels=hidden_channels, num_layers=n_layers,
                                   out_channels=out_dim, dropout=dropout, bias=bias)


        elif model_name == 'gnan':
            if not is_graph_task:
                model = GNAN(in_channels=num_features,
                             hidden_channels=hidden_channels, num_layers=n_layers,
                             out_channels=out_dim, dropout=dropout, bias=bias, device=device,
                             limited_m=one_m,
                             normalize_m=normalize_m)
            else:
                model = TensorGNAN(in_channels=num_features,
                                   hidden_channels=hidden_channels, num_layers=n_layers,
                                   out_channels=out_dim, dropout=dropout, bias=bias, device=device,
                                   limited_m=one_m,
                                   normalize_m=normalize_m, final_agg=final_agg, is_graph_task=is_graph_task,
                                   readout_n_layers=readout_n_layers)

        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('model size: {:.3f}MB'.format(size_all_mb))
        if wandb_flag:
            wandb.log({'model_size': size_all_mb})
        model.to(device)
        config = {
            'lr': lr,
            'loss': loss_type.__name__,
            'hidden_channels': hidden_channels,
            'n_conv_layers': n_layers,
            'output_dim': out_dim,
            'num_epochs': num_epochs,
            'optimizer': optimizer_type.__name__,
            'model': model.__class__.__name__,
            'device': device.type,
            'loss_thresh': loss_thresh,
            'debug': debug,
            'wd': wd,
            'bias': bias,
            'dropout': dropout,
            'seed': seed,
            'data_name': data_name,
            'unique_run_id': unique_run_id,
            'early_stop_flag': early_stop_flag,
            'num_features': num_features,
            'limited_m': one_m,
            'normalize_m': normalize_m,
            'seed index ': i,
            'is_graph_task': is_graph_task,
            'num_classes': num_classes,
            'readout_n_layers': readout_n_layers,
            'final_agg': final_agg,
            'cross_val': 'yes' if is_graph_task else 'no',
            'is_for_plot': is_for_plot,
            'fixed_normalization': True,
            'is_regression': is_regression,
            'processed_data_dir': processed_data_dir,

        }
        for name, val in config.items():
            print(f'{name}: {val}')
        if wandb_flag:
            exp_name = f'GNAM_{model.__class__.__name__}_{data_name}'
            wandb.init(project='GNAM', reinit=True, entity='GNAN',
                       settings=wandb.Settings(start_method='thread'),
                       config=config, name=exp_name)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
        loss = loss_type()
        early_stop = EarlyStopping(metric_name='Loss', patience=patience, min_is_better=True)
        best_val_acc_model_val_acc = 0
        best_val_acc_model_val_auc = 0
        best_train_loss_model_train_loss = 1000000

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.9,
            patience=100,
            verbose=True,
            min_lr=1e-8,
        )
        for epoch in range(num_epochs):
            train_loss, train_acc, train_auc = trainer.train_epoch(model, dloader=train_loader,
                                                                   loss_fn=loss,
                                                                   optimizer=optimizer,
                                                                   classify=~is_regression, device=device,
                                                                   compute_auc=compute_auc, is_graph_task=is_graph_task)
            val_loss, val_acc, val_auc = trainer.test_epoch(model, dloader=val_loader,
                                                            loss_fn=loss, classify=~is_regression,
                                                            device=device, val_mask=True, compute_auc=compute_auc,
                                                            is_graph_task=is_graph_task)

            scheduler.step(train_loss)
            if compute_auc:
                if val_auc > best_val_acc_model_val_auc:
                    best_val_acc_model_val_auc = val_auc
                    torch.save(model.state_dict(),
                               f'models/{unique_run_id}_{data_name}_{model_name}_{seed}_best_val_auc.pt')
                    best_val_auc_epoch = epoch
                    best_val_acc_model_test_loss, best_val_acc_model_test_acc, best_val_acc_model_test_auc = trainer.test_epoch(
                        model,
                        dloader=test_loader,
                        loss_fn=loss,
                        classify=~is_regression,
                        device=device, compute_auc=compute_auc, is_graph_task=is_graph_task)
                    print(f'Best Val AUC Model Test Acc: {best_val_acc_model_test_acc:.4f},'
                          f'Best Val AUC Model Test Loss: {best_val_acc_model_test_loss:.4f},'
                          f'Best Val AUC Model Val Acc: {val_acc:.4f},'
                          f'Best Val AUC Model Val Loss: {val_loss:.4f},'
                          f'Best Val AUC Model Train Acc: {train_acc:.4f},'
                          f'Best Val AUC Model Train Loss: {train_loss:.4f}')
                    if wandb_flag:
                        wandb.log({
                            'best_val_auc_model_test_acc': best_val_acc_model_test_acc,
                            'best_val_auc_model_test_loss': best_val_acc_model_test_loss,
                            'best_val_auc_epoch': best_val_auc_epoch,
                            'best_val_auc_model_val_acc': val_acc,
                            'best_val_auc_model_val_loss': val_loss,
                            'best_val_auc_model_train_acc': train_acc,
                            'best_val_auc_model_train_loss': train_loss
                        })

            else:
                if val_acc > best_val_acc_model_val_acc:
                    best_val_acc_model_val_acc = val_loss
                    torch.save(model.state_dict(),
                               f'models/{unique_run_id}_{data_name}_{model_name}_{seed}_best_val_acc.pt')
                    best_val_acc_epoch = epoch
                    best_val_acc_model_test_loss, best_val_acc_model_test_acc, _ = trainer.test_epoch(model,
                                                                                                      dloader=test_loader,
                                                                                                      loss_fn=loss,
                                                                                                      classify=~is_regression,
                                                                                                      device=device,
                                                                                                      compute_auc=compute_auc,
                                                                                                      is_graph_task=is_graph_task)

                    print(f'Best Val Acc Model Test Acc: {best_val_acc_model_test_acc:.4f},'
                          f'Best Val Acc Model Test Loss: {best_val_acc_model_test_loss:.4f},'
                          f'Best Val Acc Model Val Acc: {val_acc:.4f},'
                          f'Best Val Acc Model Val Loss: {val_loss:.4f},'
                          f'Best Val Acc Model Train Acc: {train_acc:.4f},'
                          f'Best Val Acc Model Train Loss: {train_loss:.4f}')

                    if wandb_flag:
                        wandb.log({
                            'best_val_acc_model_test_acc': best_val_acc_model_test_acc,
                            'best_val_acc_model_test_loss': best_val_acc_model_test_loss,
                            'best_val_acc_epoch': best_val_acc_epoch,
                            'best_val_acc_model_val_acc': val_acc,
                            'best_val_acc_model_val_loss': val_loss,
                            'best_val_acc_model_train_acc': train_acc,
                            'best_val_acc_model_train_loss': train_loss
                        })

            if train_loss < best_train_loss_model_train_loss:
                best_train_loss_model_train_loss = train_loss
                # best_train_loss_model_params = deepcopy(model.state_dict())
                torch.save(model.state_dict(),
                           f'models/{unique_run_id}_{data_name}_{model_name}_{seed}_best_train_loss.pt')
                best_train_loss_epoch = epoch
                best_train_loss_model_test_loss, best_train_loss_model_test_acc, best_train_loss_model_test_auc = trainer.test_epoch(
                    model,
                    dloader=test_loader,
                    loss_fn=loss,
                    classify=~is_regression,
                    device=device, compute_auc=compute_auc, is_graph_task=is_graph_task)

                print(f'Best Train Loss Model Test Acc: {best_train_loss_model_test_acc:.4f},'
                      f'Best Train Loss Model Test Loss: {best_train_loss_model_test_loss:.4f},'
                      f'Best Train Loss Model Val Acc: {val_acc:.4f},'
                      f'Best Train Loss Model Val Loss: {val_loss:.4f},'
                      f'Best Train Loss Model Train Acc: {train_acc:.4f},'
                      f'Best Train Loss Model Train Loss: {best_train_loss_model_train_loss:.4f}',
                      f'Best Train Loss Model Test AUC: {best_train_loss_model_test_auc:.4f}',
                      f'Best Train Loss Model Val AUC: {val_auc:.4f}',
                      f'Best Train Loss Model Train AUC: {train_auc:.4f}')

                if wandb_flag:
                    wandb.log({
                        'best_train_loss_model_test_acc': best_train_loss_model_test_acc,
                        'best_train_loss_model_test_loss': best_train_loss_model_test_loss,
                        'best_train_loss_epoch': best_train_loss_epoch,
                        'best_train_loss_model_val_acc': val_acc,
                        'best_train_loss_model_val_loss': val_loss,
                        'best_train_loss_model_train_acc': train_acc,
                        'best_train_loss_model_train_loss': train_loss,
                        'best_train_loss_model_test_auc': best_train_loss_model_test_auc,
                        'best_train_loss_model_val_auc': val_auc,
                        'best_train_loss_model_train_auc': train_auc
                    })

            test_loss, test_acc, test_auc = trainer.test_epoch(model, dloader=test_loader,
                                                               loss_fn=loss, classify=~is_regression,
                                                               device=device, compute_auc=compute_auc,
                                                               is_graph_task=is_graph_task)

            # model.print_m_params()
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Val Acc: {val_acc:.4f}', f'Test Loss: {test_loss:.4f}, '
                                             f'Test Acc: {test_acc:.4f}')
            early_stop(val_loss)
            if train_loss < loss_thresh:
                print(f'loss under {loss_thresh} at epoch: {epoch}')
                break
            if early_stop_flag and early_stop.early_stop:
                print(f'early stop at epoch: {epoch}')
                break
            if wandb_flag:
                wandb.log({'train_loss': train_loss,
                           'train_acc': train_acc,
                           'val_loss': val_loss,
                           'val_acc': val_acc,
                           'epoch': epoch,
                           'test_loss': test_loss,
                           'test_acc': test_acc
                           })

        # test
        test_loss, test_acc, test_auc = trainer.test_epoch(model, dloader=test_loader,
                                                           loss_fn=loss, classify=~is_regression,
                                                           device=device, compute_auc=compute_auc,
                                                           is_graph_task=is_graph_task)
        print(f'Test Loss: {test_loss:.4f}, '
              f'Test Acc: {test_acc:.4f}')

        if wandb_flag:
            wandb.log({'test_loss': test_loss,
                       'test_acc': test_acc})

            wandb.finish()


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=64)
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=3)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=1000)
    parser.add_argument('--wandb_flag', dest='wandb_flag', type=int, default=1)
    parser.add_argument('--bias', dest='bias', type=int, default=1)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0)
    parser.add_argument('--early_stop', dest='early_stop', type=int, default=0)
    parser.add_argument('--wd', dest='wd', type=float, default=0.00005)
    parser.add_argument('--data_name', dest='data_name', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'mutag', 'proteins', 'ptc_mr', 'nci1',
                                 'QM9_data_1', 'QM9_data_2', 'QM9_data_3', 'mutagenicity', 'ogb_arxiv',
                                 'tolokers'])
    parser.add_argument('--model_name', dest='model_name', type=str, default='gnan',
                        choices=['sage', 'gin', 'gatv2', 'graphconv', 'gnan'])
    parser.add_argument('--seed', dest='seed', type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--run_grid_search', dest='run_grid_search', type=int, default=0)
    parser.add_argument('--one_m', dest='one_m', type=int, default=0)
    parser.add_argument('--normalize_m', dest='normalize_m', type=int, default=1)
    parser.add_argument('--readout_n_layers', dest='readout_n_layers', type=int, default=0)
    parser.add_argument('--processed_data_dir', dest='processed_data_dir', type=str, default='processed_data')

    args = parser.parse_args()
    loss_thresh = 0.00001
    optimizer_type = torch.optim.Adam

    train_loader, val_loader, test_loader, num_features, num_classes, is_graph_task, is_regression, compute_auc = datasets.get_data(
        args.data_name, args.processed_data_dir)

    if is_regression:
        loss_type = torch.nn.MSELoss
        out_dim = 1
    else:
        if num_classes == 2:
            loss_type = torch.nn.BCEWithLogitsLoss
            out_dim = 1
        else:
            loss_type = torch.nn.CrossEntropyLoss
            out_dim = num_classes
    unique_run_id = uuid.uuid1()

    run_exp(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, num_features=num_features,
            n_layers=args.n_layers, early_stop_flag=args.early_stop, dropout=args.dropout,
            model_name=args.model_name,
            num_epochs=args.num_epochs, wandb_flag=args.wandb_flag, wd=args.wd,
            hidden_channels=args.hidden_channels, lr=args.lr, bias=args.bias, patience=args.patience,
            debug=args.debug, loss_thresh=loss_thresh, seeds=seeds, data_name=args.data_name,
            unique_run_id=unique_run_id, one_m=args.one_m, normalize_m=args.normalize_m,
            is_graph_task=is_graph_task, num_classes=num_classes,
            final_agg=args.final_agg, readout_n_layers=args.readout_n_layers, out_dim=out_dim,
            is_for_plot=args.is_for_plot, is_regression=is_regression, processed_data_dir=args.processed_data_dir,
            compute_auc=compute_auc)
