import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def get_accuracy(outputs, labels):

    if outputs.dim() == 2 and outputs.shape[-1] > 1:
        return get_multiclass_accuracy(outputs, labels)
    else:
        y_prob = torch.sigmoid(outputs).view(-1)
        y_prob = y_prob > 0.5
        return (labels == y_prob).sum().item()

def get_multiclass_accuracy(outputs, labels):
    assert outputs.size(1) >= labels.max().item() + 1
    probas = torch.softmax(outputs, dim=-1)
    preds = torch.argmax(probas, dim=-1)
    correct = (preds == labels).sum()
    acc = correct
    return acc


def train_epoch(model, dloader, loss_fn, optimizer, device, classify=True, label_index=0, compute_auc=False, is_graph_task=True):
    with torch.autograd.set_detect_anomaly(True):
        running_loss = 0.0
        n_samples = 0
        all_probas = np.array([])
        all_labels = np.array([])
        if classify:
            running_acc = 0.0
        for i, data in enumerate(dloader):
            if len(data.y.shape) > 1:
                labels = data.y[:, label_index].view(-1, 1).flatten()
                labels = labels.float()
            else:
                labels = data.y.flatten()
            if -1 in labels:
                labels = (labels + 1) / 2
            if loss_fn.__class__.__name__ == 'CrossEntropyLoss':
                labels = labels.long()

            non_zero_ids = None
            if model.__class__.__name__ == 'GNAM':
                    labels = labels[data.train_mask]
                    non_zero_ids = torch.nonzero(data.train_mask).flatten()
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if non_zero_ids is not None:
                outputs = model.forward(data, non_zero_ids)
            else:
                outputs = model.forward(data)
                if not is_graph_task:
                    labels = labels[data.train_mask]
                    outputs = outputs[data.train_mask]

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            n_samples += len(labels)
            if outputs.dim() == 2 and outputs.shape[-1] == 1:
                loss = loss_fn(outputs.flatten(), labels.float())
            else:
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            if compute_auc:
                probas = torch.sigmoid(outputs).view(-1)
                all_probas = np.concatenate((all_probas, probas.detach().cpu().numpy()))
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()))
            running_loss += loss.item()

            if classify:
                running_acc += get_accuracy(outputs, labels)

        if compute_auc:
            auc = roc_auc_score(all_labels, all_probas)

        if classify:
            if compute_auc:
                return running_loss / len(dloader), running_acc / n_samples, auc
            else:
                return running_loss / len(dloader), running_acc / n_samples, -1
        else:
            return running_loss / len(dloader), -1


def test_epoch(model, dloader, loss_fn, device, classify=True, label_index=0, compute_auc=False, val_mask=False, is_graph_task=True):
    with torch.no_grad():
        running_loss = 0.0
        all_probas = np.array([])
        all_labels = np.array([])
        n_samples = 0
        if classify:
            running_acc = 0.0
        model.eval()
        for i, data in enumerate(dloader):
            if len(data.y.shape) > 1:
                labels = data.y[:, label_index].view(-1, 1).flatten()
                labels = labels.float()
            else:
                labels = data.y.flatten()
            if -1 in labels:
                labels = (labels + 1) / 2
            if loss_fn.__class__.__name__ == 'CrossEntropyLoss':
                labels = labels.long()
            inputs = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            non_zero_ids = None
            if model.__class__.__name__ == 'GNAM':
                if val_mask:
                    labels = labels[data.val_mask]
                    non_zero_ids = torch.nonzero(data.val_mask).flatten()
                else:
                    labels = labels[data.test_mask]
                    non_zero_ids = torch.nonzero(data.test_mask).flatten()

            # forward
            if non_zero_ids is not None:
                outputs = model.forward(inputs, non_zero_ids)
            else:
                outputs = model.forward(inputs)
                if not is_graph_task:
                    if val_mask:
                        outputs = outputs[data.val_mask]
                        labels = labels[data.val_mask]
                    else:
                        outputs = outputs[data.test_mask]
                        labels = labels[data.test_mask]
            n_samples += len(labels)
            if outputs.dim() == 2 and outputs.shape[-1] == 1:
                loss = loss_fn(outputs.flatten(), labels.float())
            else:
                loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            if classify:
                running_acc += get_accuracy(outputs, labels)
            if compute_auc:
                probas = torch.sigmoid(outputs).view(-1)
                all_probas = np.concatenate((all_probas, probas.detach().cpu().numpy()))
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()))

        if compute_auc:
            auc = roc_auc_score(all_labels, all_probas)
        if classify:
            if compute_auc:
                return running_loss / len(dloader), running_acc / n_samples, auc
            else:
                return running_loss / len(dloader), running_acc / n_samples, -1
        else:
            return running_loss / len(dloader), -1

