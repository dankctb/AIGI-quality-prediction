import os
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

from datasets.AI1k_Augmented import AI1k_load_data
from utils.compute_metrics_Jean import compute_metrics
from models.MetaIQA_Augmented import MetaIQA

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=10):
    """Decay learning rate by a factor of decay_rate every lr_decay_epoch epochs."""

    decay_rate = 0.8**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def save_data(gt, pred, epoch: int, split: int, datatype: str, data_dir: str,):
    np.save(os.path.join(data_dir, f'{datatype}_split_{split}_epoch_{epoch}_ground_truth.npy'), gt)
    np.save(os.path.join(data_dir, f'{datatype}_split_{split}_epoch_{epoch}_predictions.npy'), pred)
         

def plot_metrics(plots_dir, exp_name, train_metrics, val_metrics, metric_name, split):
    plt.figure()
    plt.plot(train_metrics, label='Train ' + metric_name)
    plt.plot(val_metrics, label='Validation ' + metric_name)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{exp_name} - Split {split}')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'split_{split}_{metric_name}.jpg'))
    plt.close()

def plot_average_metrics(plots_dir, exp_name, avg_train_metrics, avg_val_metrics, metric_name):
    plt.figure()
    plt.plot(avg_train_metrics, label='Average Train ' + metric_name)
    plt.plot(avg_val_metrics, label='Average Validation ' + metric_name)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{exp_name} - Average')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'avg_{metric_name}.jpg'))
    plt.close()


def evaluate_model(
    split_type: str,
    split: int,
    model: nn.Module,
    criterion: nn.Module,
    fetch_example,
) -> tuple:
    dataloader_valid = AI1k_load_data(split_type, split)
    model.eval()

    ground_truth = []
    predicted = []
    losses = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader_valid):
            inputs, labels = fetch_example(data)

            model_outputs = model(*inputs)
            loss = criterion(model_outputs, labels)
            losses.append(loss.float().cpu())
            ground_truth.extend(labels.float().cpu().tolist())
            predicted.extend(model_outputs.float().cpu().tolist())
    
    # loss = np.mean(losses)
    ground_truth = np.array(ground_truth)[:,0]
    predictions = np.array(predicted)[:,0]
    return losses, ground_truth, predictions

def finetune_model(FusionModel, fetch_example, exp_name, num_epochs = 100, num_splits = 5):
    plots_dir = os.path.join('AI-1k', "results", exp_name, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    data_dir = os.path.join('AI-1k', "results", exp_name, "data", "Jean")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plcc_list = []
    srocc_list = []
    mse_list = []

    avg_train_plcc_list = np.zeros(num_epochs)
    avg_train_srocc_list = np.zeros(num_epochs)
    avg_train_mse_list = np.zeros(num_epochs)
    avg_val_plcc_list = np.zeros(num_epochs)
    avg_val_srocc_list = np.zeros(num_epochs)
    avg_val_mse_list = np.zeros(num_epochs)
    
    for split in range(num_splits):
        model = FusionModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

        with torch.cuda.device(0):
            model.cuda()

            best_spearman = 0
            best_pearson = 0
            best_mse = float('inf')

            train_plcc_list = []
            train_srocc_list = []
            train_mse_list = []
            val_plcc_list = []
            val_srocc_list = []
            val_mse_list = []

            for epoch in range(num_epochs):
                optimizer = exp_lr_scheduler(optimizer, epoch)

                dataloader_train = AI1k_load_data('train', split)
                model.train()

                for batch_idx, data in enumerate(dataloader_train):
                    inputs, labels = fetch_example(data)

                    optimizer.zero_grad()
                    outputs = model(*inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                train_losses, train_gt, train_pred = evaluate_model('train', split, model, criterion, fetch_example)
                save_data(train_gt, train_pred, epoch, split, "train", data_dir)
                train_spearman, train_pearson, train_mse = compute_metrics(train_losses, train_gt, train_pred)
                train_plcc_list.append(train_pearson)
                train_srocc_list.append(train_spearman)
                train_mse_list.append(train_mse)
                print(f'Train Results - Split: {split}, Epoch: {epoch}, PLCC: {train_pearson:.4f}, SROCC: {train_spearman:.4f}, MSE: {train_mse:.4f}')
                
                
                val_losses, val_gt, val_pred = evaluate_model('val', split, model, criterion, fetch_example)
                save_data(val_gt, val_pred, epoch, split, "val", data_dir)
                val_spearman, val_pearson, val_mse = compute_metrics(val_losses, val_gt, val_pred)
                val_plcc_list.append(val_pearson)
                val_srocc_list.append(val_spearman)
                val_mse_list.append(val_mse)

                best_spearman = max(best_spearman, val_spearman)
                best_pearson = max(best_pearson, val_pearson)
                best_mse = min(best_mse, val_mse)
                    
                print(f'Validation Results - Split: {split}, Epoch: {epoch}, PLCC: {val_pearson:.4f}, SROCC: {val_spearman:.4f}, MSE: {val_mse:.4f}, '
                      f'Best PLCC: {best_pearson:.4f}, Best SROCC: {best_spearman:.4f}, Best MSE: {best_mse:.4f}')

            plcc_list.append(best_pearson)
            srocc_list.append(best_spearman)
            mse_list.append(best_mse)

            # Save plots
            plot_metrics(plots_dir, exp_name, train_plcc_list, val_plcc_list, 'PLCC', split)
            plot_metrics(plots_dir, exp_name, train_srocc_list, val_srocc_list, 'SROCC', split)
            plot_metrics(plots_dir, exp_name, train_mse_list, val_mse_list, 'MSE', split)
            
            # Update the average metric lists
            avg_train_plcc_list += np.array(train_plcc_list)
            avg_train_srocc_list += np.array(train_srocc_list)
            avg_train_mse_list += np.array(train_mse_list)
            avg_val_plcc_list += np.array(val_plcc_list)
            avg_val_srocc_list += np.array(val_srocc_list)
            avg_val_mse_list += np.array(val_mse_list)
            
    # At the end of the loop through all splits, calculate the average metrics
    avg_train_plcc_list /= num_splits
    avg_train_srocc_list /= num_splits
    avg_train_mse_list /= num_splits
    avg_val_plcc_list /= num_splits
    avg_val_srocc_list /= num_splits
    avg_val_mse_list /= num_splits

    # Save plots
    plot_average_metrics(plots_dir, exp_name, avg_train_plcc_list, avg_val_plcc_list, 'PLCC')
    plot_average_metrics(plots_dir, exp_name, avg_train_srocc_list, avg_val_srocc_list, 'SROCC')
    plot_average_metrics(plots_dir, exp_name, avg_train_mse_list, avg_val_mse_list, 'MSE')

    print(f'Average PLCC: {np.mean(plcc_list):.4f}')
    print(f'Average SROCC: {np.mean(srocc_list):.4f}')
    print(f'Average MSE: {np.mean(mse_list):.4f}')
    
    np.save(os.path.join(data_dir, 'split_best_plcc.npy'), plcc_list)
    np.save(os.path.join(data_dir, 'split_best_srocc.npy'), srocc_list)
    np.save(os.path.join(data_dir, 'split_best_mse.npy'), mse_list)
    np.save(os.path.join(data_dir, 'avg_train_plcc_list.npy'), avg_train_plcc_list)
    np.save(os.path.join(data_dir, 'avg_train_srocc_list.npy'), avg_train_srocc_list)
    np.save(os.path.join(data_dir, 'avg_train_mse_list.npy'), avg_train_mse_list)
    np.save(os.path.join(data_dir, 'avg_val_plcc_list.npy'), avg_val_plcc_list)
    np.save(os.path.join(data_dir, 'avg_val_srocc_list.npy'), avg_val_srocc_list)
    np.save(os.path.join(data_dir, 'avg_val_mse_list.npy'), avg_val_mse_list)
    

    # # Final evaluation on test set
    # dataloader_test = AI1k_load_data('test')
    # model.eval()
    # test_spearman, test_pearson, test_mse = compute_metrics(dataloader_test, model, num_epochs - 1, criterion, fetch_example)
    # print(f'Test Results - PLCC: {test_pearson:.4f}, SROCC: {test_spearman:.4f}, MSE: {test_mse:.4f}')








