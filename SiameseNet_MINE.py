# from torchvision.datasets import FashionMNIST
import torch
from torchvision import transforms

from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

from Data_loader_MINE_siamese import Custom_Dataloader, Rescale, RandomCrop, ToTensor

from trainer_siamese import fit
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from losses import TripletLoss
from networks import EmbeddingNet_MINE, EmbeddingNet, TripletNet, MNAS_Net, MNAS_Net_transformed, ClassificationNet, SiameseNet_MINE
from metrics import AccumulatedAccuracyMetric

cuda = torch.cuda.is_available()

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
