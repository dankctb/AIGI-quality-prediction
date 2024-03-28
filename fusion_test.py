
import os
import math
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.preprocessing import normalize
from einops import rearrange, repeat

# from datasets.AI1k import AI1k_load_data
# from utils.compute_metrics import compute_metrics
from utils.finetune_model_Jean import finetune_model
from models.MetaIQA import MetaIQA
from models.CNNIQAnet import CNNIQAnet
#from models.BRISQUE import BRISQUE

num_epochs = 25
num_splits = 10


exp_name = "CLIP_features_2fc_Meta_AI1k-(ViT-16)-mean"

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        
        # Load the pretrained ResNet18 model
        self.quality_model = MetaIQA('model_IQA/TID2013_KADID10K_IQA_Meta_resnet18.pt')

        self.fc_clip_image = nn.Linear(512, 512)
        self.fc_clip_text = nn.Linear(512, 512)
        self.fc_clip_text.weight.data.normal_(0, 0.0)
        self.fc_clip_text.bias.data.zero_()
        
        self.relu_clip_feat = nn.PReLU()
        self.drop_clip_feat = nn.Dropout(p=0.5)

        self.fc_final = nn.Linear(1024, 1024)
        # self.sig = nn.Sigmoid()
        # self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=1)

        # Define the fully connected layers
        input_size = 1536
        intermediate_size = 512
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, intermediate_size)
        self.fc3 = nn.Linear(intermediate_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, clip_image, clip_text):
        # Pass the input through the pretrained model
        
        quality, quality_features = self.quality_model(x)
        #print("Error flag --------------------")#not through this yet

        image_out = self.fc_clip_image(clip_image)
        # # similarity_features = self.bn_clip_feat(similarity_features)
        image_out = self.relu_clip_feat(image_out)
        image_out = self.drop_clip_feat(image_out)

        text_out = self.fc_clip_text(clip_text)
        # # similarity_features = self.bn_clip_feat(similarity_features)
        text_out = self.relu_clip_feat(text_out)
        text_out = self.drop_clip_feat(text_out)

        combined = torch.cat((image_out, text_out), dim=1)
        fc_final = self.fc_final(combined)

        combined_1 = torch.cat((quality_features, fc_final), dim=1)


        
        # Pass the concatenated tensor through the fully connected layer
        x = self.fc1(combined_1)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.sigmoid(x)
        # return x
    
def fetch_example(data):
    inputs = data['image']
    clip_image = data['clip_image_features']
    clip_text = data['clip_text_features']
    batch_size = inputs.size()[0]
    labels = data['rating'].view(batch_size, -1) / 5.0
    
    inputs, clip_image, clip_text, labels = Variable(inputs.float().cuda()), Variable(clip_image.float().cuda()),Variable(clip_text.float().cuda()), Variable(labels.float().cuda())
    return [inputs, clip_image, clip_text], labels

finetune_model(FusionModel, fetch_example, exp_name, num_epochs = num_epochs, num_splits = num_splits)
