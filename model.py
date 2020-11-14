import torch.nn as nn
import torch
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.conv1_ = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2_ = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

    def forward(self, x):
        out = F.relu(self.conv1_(x))
        out = F.relu(self.conv2_(out))
        out = x + out

        return out

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2,2))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.res_layers = nn.ModuleList()

        for i in range(3):
            self.res_block = ResnetBlock()
            self.res_layers.append(self.res_block)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv_out = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(1, 1))

        self.fc1 = nn.Linear(in_features=2304, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, img):
        out = self.conv1(img)
        out = F.relu(out, inplace=True)
        out = self.max_pool1(out)
        for i in range(3):
            out = self.res_layers[i](out)

        out = self.conv2(out)
        out = F.relu(out, inplace=True)
        out = self.max_pool2(out)
        out = F.relu(self.conv_out(out))
        out = torch.reshape(out, [out.size(0), -1])   # batch x 2304
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        logits = self.fc3(out)
        return logits


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        loss = self.ce_loss(prediction, target)
        return loss
