import torch
import numpy as np
import os

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, model_save_path,
        metrics=[], start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            

        print(message)
        torch.save(model.embedding_net.state_dict(), os.path.join(model_save_path, "weights_embedding_net_"+str(epoch)+".pt"))


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        if cuda:
            data = tuple(d.type(torch.cuda.FloatTensor).cuda() for d in data)
            if target is not None:
                target = target.type(torch.cuda.FloatTensor).cuda() 

        optimizer.zero_grad()
        outputs = model(*data)

        loss_outputs = loss_fn(outputs, target)

        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)
        print("train_batch_idx", batch_idx)

        message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            batch_idx * len(data[0]), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), np.mean(losses))
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.type(torch.cuda.FloatTensor).cuda() for d in data)
                # data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.type(torch.cuda.FloatTensor).cuda()

            outputs = model(*data)

            loss_outputs = loss_fn(outputs, target)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)
            print("val_batch_index", batch_idx)

    return val_loss, metrics
