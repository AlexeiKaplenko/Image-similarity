import torch
import numpy as np
import os

n_patches = 64
batch_size = 1
patch_size = 224

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
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, log_interval, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            
        print(message)
        # torch.save(model.module.embedding_net.state_dict(), os.path.join(model_save_path, "weights_embedding_net_"+str(epoch)+".pt")) # if 2 GPUS
        torch.save(model.state_dict(), os.path.join(model_save_path, "weights_embedding_net_"+str(epoch)+".pt")) #if 1 GPU


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        if cuda:

            data = tuple(d.type(torch.cuda.FloatTensor).cuda().view(-1, 3, patch_size, patch_size) for d in data)

            target = target.type(torch.cuda.FloatTensor).cuda()

        optimizer.zero_grad()
        outputs = model(*data)
        outputs = outputs.view(batch_size, n_patches, -1).mean(1) # avg over crops

        loss_outputs = loss_fn(outputs, target)

        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        #Accuracy
        outputs = (outputs>0.5).float()
        correct = (outputs == target).float().sum()
        accuracy = correct/outputs.shape[0]

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx, len(train_loader.dataset),

                100. * batch_idx / len(train_loader), np.mean(losses))

            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, log_interval, metrics):

    losses = []
    val_loss = 0

    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(val_loader):

            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.type(torch.cuda.FloatTensor).cuda().view(-1, 3, patch_size, patch_size) for d in data)

                target = target.type(torch.cuda.FloatTensor).cuda()

            outputs = model(*data)
            outputs = outputs.view(batch_size, n_patches, -1).mean(1) # avg over crops
            loss_outputs = loss_fn(outputs, target)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

            #Accuracy
            outputs = (outputs>0.5).float()
            correct = (outputs == target).float().sum()
            accuracy = correct/outputs.shape[0]


            if batch_idx % log_interval == 0:
                message = 'Val: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx, len(val_loader.dataset),
                100. * batch_idx / len(val_loader), np.mean(losses))

                for metric in metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())

                print(message)
                losses = []

    return val_loss, metrics
