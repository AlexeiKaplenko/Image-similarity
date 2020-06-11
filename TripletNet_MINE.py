# from torchvision.datasets import FashionMNIST
import torch
from torchvision import transforms

from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

from Data_loader_MINE_triplets import Custom_Dataloader, Rescale, RandomCrop, ToTensor


from trainer_triplet import fit
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt

from losses import TripletLoss, TripletLoss_log1p, TripletLoss_64_patches
from networks import EmbeddingNet_MINE, EmbeddingNet, TripletNet, TripletNet_MINE, MNAS_Net, MNAS_Net_transformed, ClassificationNet, SiameseNet_MINE
from metrics import AccumulatedAccuracyMetric

margin = 1.
loss_fn = TripletLoss_64_patches(margin)

embedding_net = MNAS_Net_transformed()
model = TripletNet_MINE(embedding_net)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

if cuda:
    model.cuda()


lr = 0.1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 2, gamma=0.9, last_epoch=-1)
n_epochs = 16
log_interval = 1000

batch_size = 1

n_classes = 2

classes = ['Similar', 'Non-Similar']
colors = ['#1f77b4', '#ff7f0e']


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(2):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

fit(model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, './weights')



