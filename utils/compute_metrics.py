import os
import torch
import pandas as pd
import numpy as np
import warnings
from PIL import Image
import csv

from torch import nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")



def compute_metrics(
    validation_dataloader: DataLoader,
    model: nn.Module,
    epoch: int,
    split: int,
    datatype: str,
    exp_name: str,
    criterion: nn.Module,
    fetch_example,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> tuple:
    ground_truth_ratings = []
    predicted_ratings = []
    batch_losses = []
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_dataloader):
            inputs, labels = fetch_example(data)

            model_outputs = model(*inputs)
            loss = criterion(model_outputs, labels)
            batch_losses.append(loss.float().cpu())
            ground_truth_ratings.extend(labels.float().cpu().tolist())
            predicted_ratings.extend(model_outputs.float().cpu().tolist())

    loss = np.mean(batch_losses)
    ground_truth = np.array(ground_truth_ratings)[:,0]
    predictions = np.array(predicted_ratings)[:,0]
    
    data_dir = os.path.join('AI-1k')
    np.save(os.path.join(data_dir, "scatter_plots", "Trinh", exp_name, datatype + '_split_' + str(split) + 'epoch_' + str(epoch) + 'ground_truth.npy'), ground_truth)
    np.save(os.path.join(data_dir, "scatter_plots", "Trinh", exp_name, datatype + '_split_' + str(split) + 'epoch_' + str(epoch) + 'predictions.npy'), predictions)

#     ground_truth_pd = pd.DataFrame(ground_truth_ratings)
#     predictions_pd = pd.DataFrame(predicted_ratings)
    
#     ground_truth_pd.to_csv("myfile.csv", mode='a')
#     predictions_pd.to_csv("myfile.csv", mode='a')        
    
    spearman_coefficient = spearmanr(ground_truth, predictions)[0]
    pearson_coefficient = pearsonr(ground_truth, predictions)[0]
    

    # Save ground_truth_ratings and predicted_ratings (optional)
    # np.save(f'ground_truth_ratings.npy', ground_truth_ratings_array)
    # np.save(f'predicted_ratings_{epoch:02d}.npy', predicted_ratings_array)

    return spearman_coefficient, pearson_coefficient, loss




































# import os
# import torch
# import pandas as pd
# import numpy as np
# import warnings
# from PIL import Image

# from torch import nn
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
# from torchvision import transforms, models
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.dataloader import default_collate

# from sklearn.model_selection import train_test_split
# from scipy.stats import spearmanr, pearsonr
# import torch.optim as optim

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# warnings.filterwarnings("ignore")

# def compute_metrics(
#     validation_dataloader: DataLoader,
#     model: nn.Module,
#     epoch: int,
#     criterion: nn.Module,
#     fetch_example,
#     device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
# ) -> tuple:
#     ground_truth_ratings = []
#     predicted_ratings = []
#     batch_losses = []
#     with torch.no_grad():
#         for batch_idx, data in enumerate(validation_dataloader):
#             inputs, labels = fetch_example(data)

#             model_outputs = model(*inputs)
#             loss = criterion(model_outputs, labels)
#             batch_losses.append(loss.float().cpu())
#             ground_truth_ratings.extend(labels.float().cpu().tolist())
#             predicted_ratings.extend(model_outputs.float().cpu().tolist())

#     loss = np.mean(batch_losses)
#     ground_truth = np.array(ground_truth_ratings)[:,0]
#     predictions = np.array(predicted_ratings)[:,0]
#     spearman_coefficient = spearmanr(ground_truth, predictions)[0]
#     pearson_coefficient = pearsonr(ground_truth, predictions)[0]
    

#     # Save ground_truth_ratings and predicted_ratings (optional)
#     # np.save(f'ground_truth_ratings.npy', ground_truth_ratings_array)
#     # np.save(f'predicted_ratings_{epoch:02d}.npy', predicted_ratings_array)

#     return spearman_coefficient, pearson_coefficient, loss



