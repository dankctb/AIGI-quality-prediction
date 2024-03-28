import os
import math
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import ShuffleSplit, train_test_split
from scipy.stats import spearmanr, pearsonr

from datasets.AI1k import AI1k_load_data
from models.MetaIQA import MetaIQA

def compute_metrics(
    validation_dataloader: DataLoader,
    model: nn.Module,
    epoch: int,
    criterion: nn.Module,
    fetch_example,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple:
    ground_truth_ratings = []
    predicted_ratings = []
    batch_losses = []
    predictions = []
    preactivations = []
    features = []
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_dataloader):
            inputs, labels = fetch_example(data)

            batch_predictions = model(*inputs)
            
            predictions.append(batch_predictions.cpu())
            loss = criterion(batch_predictions, labels)
            batch_losses.append(loss.float().cpu())
            ground_truth_ratings.extend(labels.float().cpu().tolist())

    loss = np.mean(batch_losses)
    ground_truth = np.array(ground_truth_ratings)[:,0]
    predictions = np.vstack(predictions)[:,0]

    spearman_coefficient = spearmanr(ground_truth, predictions)[0]
    pearson_coefficient = pearsonr(ground_truth, predictions)[0]
    
    print("predictions.shape", predictions.shape)


    return spearman_coefficient, pearson_coefficient, loss, predictions, preactivations, features

def eval_model(FusionModel, fetch_example, data_dir = os.path.join('test_AI1k')):
    model = FusionModel(768)
    criterion = nn.MSELoss()

    with torch.cuda.device(0):
        criterion = nn.MSELoss()
        model.cuda()

        dataloader_all_eval = AI1k_load_data('all')
        model.eval()

        all_spearman, all_pearson, all_mse, predictions, preactivations, features = compute_metrics(dataloader_all_eval, model, 0, criterion, fetch_example)
        print(f'All Results - PLCC: {all_pearson:.4f}, SROCC: {all_spearman:.4f}, MSE: {all_mse:.4f}')

    # Save ground_truth_ratings and predicted_ratings (optional)
    np.save(os.path.join(data_dir, f'all_MetaIQA_predictions.npy'), predictions)
    np.save(os.path.join(data_dir, f'all_MetaIQA_preactivations.npy'), preactivations)
    np.save(os.path.join(data_dir, f'all_MetaIQA_features.npy'), features)

class FusionModel(nn.Module):
    def __init__(self, new_input_size):
        super(FusionModel, self).__init__()
        
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
        # Concatenate the final prediction output and the new input
        print("features shape in forward : ",features.shape)
        print("similarity_features shape in forward : ",similarity_features.shape)
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

def fetch_example(data):
    inputs = data['image']
    print("image shape in fetch_example : ",inputs.shape)
    similarity = data['similarity'].squeeze()
    print("similarity shape in fetch_example : ",similarity.shape)
    batch_size = inputs.size()[0]
    labels = data['rating'].view(batch_size, -1) / 5.0
    
    inputs, similarity, labels = Variable(inputs.float().cuda()), Variable(similarity.float().cuda()), Variable(labels.float().cuda())
    return [inputs, similarity], labels

eval_model(FusionModel, fetch_example)
