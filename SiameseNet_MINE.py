# from torchvision.datasets import FashionMNIST
import torch
from torchvision import transforms

from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
# from torchsummary import summary

from Data_loader_MINE_siamese import Custom_Dataloader, Rescale, RandomCrop, ToTensor

from trainer_siamese import fit
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt

from losses import TripletLoss
from networks import EmbeddingNet_MINE, EmbeddingNet, TripletNet, MNAS_Net, MNAS_Net_transformed, ClassificationNet, SiameseNet_MINE
from metrics import AccumulatedAccuracyMetric
# from model import EfficientNet


batch_size = 1

train_dataset_transformed = Custom_Dataloader(train=True, train_root_dir='/data/datasets/FULL_dataset_1/train', \
                transform=transforms.Compose([
                # Rescale((2048,2048)),
                RandomCrop((224,224)),
                ToTensor()
                ]))

val_dataset_transformed = Custom_Dataloader(train=False, val_root_dir='/data/datasets/FULL_dataset_1/val', \
                transform=transforms.Compose([
                # Rescale((2048,2048)),
                RandomCrop((224,224)),
                ToTensor()
                ]))

cuda = torch.cuda.is_available()
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset_transformed, batch_size=batch_size, shuffle=False, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset_transformed, batch_size=batch_size, shuffle=False, **kwargs)

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

margin = 1.

embedding_net = MNAS_Net_transformed()
model = SiameseNet_MINE(embedding_net)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

#if 2 GPUS
# model.module.embedding_net.load_state_dict(torch.load('/home/oleksii/JUPYTER_SHARED/PROJECTS/Image_similarity/siamese-triplet/weights/weights_embedding_net_12_(copy).pt'))

#if 1 GPU
model.embedding_net.load_state_dict(torch.load('/home/oleksii/JUPYTER_SHARED/PROJECTS/Image_similarity/siamese-triplet/weights/SiameseNet_01.pt'))

if cuda:
    model.cuda()

# Freeze pre-trained layers for embedding extraction
# for layer in model.module.embedding_net.parameters():
#     layer.requires_grad = False

# for layer_n, layer in enumerate(model.module.embedding_net.parameters()): # if 2 GPUS
for layer_n, layer in enumerate(model.embedding_net.parameters()):
    # model.module.embedding_net.parameters() # if 2 GPUS
    model.embedding_net.parameters() # if 1 GPU
    model.embedding_net.parameters()

    if layer_n < 155:
        layer.requires_grad = False


loss_fn = nn.BCELoss() 
lr = 0.5e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.5, last_epoch=-1)
n_epochs = 100
log_interval = 1000

fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, './weights', metrics=[AccumulatedAccuracyMetric()])
