import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNet_MINE(nn.Module):
    def __init__(self):
        super(EmbeddingNet_MINE, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5, padding = 8), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5, padding = 8), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 128, 3, padding = 4), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(128, 256, 3, padding = 4), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class MNAS_Net(nn.Module):
    def __init__(self):
        super(MNAS_Net, self).__init__()
        self.convnet = torchvision.models.mnasnet1_0(pretrained=False, progress=True)
        self.convnet.layers[0] = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=2, bias=True)
        self.convnet.classifier = nn.Linear(1280, 1024)

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class MNAS_Net_transformed(nn.Module):
    def __init__(self):
        super(MNAS_Net_transformed, self).__init__()
        self.convnet = torchvision.models.mnasnet1_0(pretrained=True, progress=True)
        self.convnet.classifier = nn.Linear(1280, 256)

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def get_embedding(self, x):
        return self.forward(x)





class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class ClassificationNet_MINE(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

class SiameseNet_MINE(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet_MINE, self).__init__()
        self.embedding_net = embedding_net
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)

        difference_embeddings = torch.abs((output1 - output2)**2)

        output = self.fc1(difference_embeddings)
        output = self.fc2(output)

        sigmoid = nn.Sigmoid()
        scores = sigmoid(output)

        return scores


    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletNet_MINE(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet_MINE, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
