import torch
import numpy as np
import os
from torchvision import transforms
from Data_loader_MINE_triplets import Custom_Dataloader, Rescale, RandomCrop, ToTensor


n_patches = 64
batch_size = 1
patch_size = 224

def fit(model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, model_save_path,
        metrics=[], start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(start_epoch, n_epochs, 1):
        scheduler.step()
        print("epoch", epoch)

        train_dataset_transformed = Custom_Dataloader(train=True,n_epochs=n_epochs, current_epoch=epoch, train_root_dir='/data/datasets/FULL_dataset_1/train', \
                        transform=transforms.Compose([
                        # Rescale((2048,2048)),
                        RandomCrop((224,224)),
                        ToTensor()
                        ]))

        val_dataset_transformed = Custom_Dataloader(train=False, n_epochs=n_epochs, current_epoch=epoch, val_root_dir='/data/datasets/FULL_dataset_1/val', \
                    transform=transforms.Compose([
                    # Rescale((2048,2048)),
                    RandomCrop((224,224)),
                    ToTensor()
                    ]))

        cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}
        train_loader = torch.utils.data.DataLoader(train_dataset_transformed, batch_size=batch_size, shuffle=False, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset_transformed, batch_size=batch_size, shuffle=False, **kwargs)


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
        torch.save(model.module.embedding_net.state_dict(), os.path.join(model_save_path, "weights_embedding_net_"+str(epoch)+".pt"))


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data = tuple(d.type(torch.cuda.FloatTensor).cuda().view(-1, 3, patch_size, patch_size) for d in data)
            if type(target) is int:
                target = target.type(torch.cuda.FloatTensor).cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        loss_outputs = loss_fn(*outputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, loss_outputs)

        if batch_idx % log_interval == 0:
            print("train_batch_idx", batch_idx)
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
            # target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.type(torch.cuda.FloatTensor).cuda().view(-1, 3, patch_size, patch_size) for d in data)

                if type(target) is int:
                    target = target.type(torch.cuda.FloatTensor).cuda()
            outputs = model(*data)
 
            loss_outputs = loss_fn(*outputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, loss_outputs)

            if batch_idx % log_interval == 0:
                print("val_batch_idx", batch_idx)
                message = 'Val: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx, len(val_loader.dataset),
                    100. * batch_idx / len(val_loader), np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())

                print(message)
                losses = []

    return val_loss, metrics
