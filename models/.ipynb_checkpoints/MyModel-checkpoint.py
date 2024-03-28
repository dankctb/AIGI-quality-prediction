from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
import csv

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr, pearsonr
use_gpu = True

from .MetaIQA import MetaIQA

class MyModel(nn.Module):
    def __init__(self, new_input_size):
        super(MyModel, self).__init__()
        
        # Load the pretrained ResNet18 model
        self.pretrained_model = MetaIQA('model_IQA/TID2013_KADID10K_IQA_Meta_resnet18.pt')
        input_size = 512+new_input_size
        intermediate_size = 512
        
        # Define the fully connected layer
        self.fc1 = nn.Linear(input_size, input_size)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(input_size, intermediate_size)
        self.bn2 = nn.BatchNorm1d(intermediate_size)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(intermediate_size, 1)
        
    def forward(self, x, similarity_features):
        # Pass the input through the pretrained model and get the final prediction output
        x, features = self.pretrained_model(x)
        
        # Pass the new input through the new input layer
        # new_x = self.new_input(new_x)
        
        # Concatenate the final prediction output and the new input
        x = torch.cat((features, similarity_features), dim=1)
        
        # Pass the concatenated tensor through the fully connected layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        return x

