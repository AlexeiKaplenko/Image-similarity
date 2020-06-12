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

fit(model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, './weights')



