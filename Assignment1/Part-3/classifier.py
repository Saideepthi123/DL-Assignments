mport torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class SimpleClassifier(nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Classifier(nn.Module):
    class Classifier(nn.Module):
     # TODO: implement me
     def init(self):
         super(Classifier, self).init()
         self.features = nn.Sequential(nn.Conv2d(3, 64, 3),
                                    nn.Conv2d(64, 64, 3),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(64, 128, 3),
                                    nn.Conv2d(128, 128, 3),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(128, 256, 3),
                                    nn.Conv2d(256, 256, 3),
                                    nn.Conv2d(256, 256, 3),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(256, 512, 3),
                                    nn.Conv2d(512, 512, 3),
                                    nn.Conv2d(512, 512, 3),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(512, 512, 3),
                                    nn.Conv2d(512, 512, 3),
                                    nn.Conv2d(512, 512, 3),
                                    )
         
    
         self.para = nn.Sequential(
             nn.Dropout(),
             nn.ReLU(inplace=True),
             nn.Dropout(),
             nn.Linear(4096, 4096),
             nn.ReLU(inplace=True),
             nn.Linear(4096, NUM_CLASSES),
         )
   

     def forward(self, x):
         x = self.features(x)
         x = torch.flatten(x)
         x = self.para(x)
         return x