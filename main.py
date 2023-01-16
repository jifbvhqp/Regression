# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# For data preprocess
import numpy as np
import csv
import os
# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from data_loader import prep_dataloader
from network import NeuralNet
from run import train,test,dev
from plot import plot_learning_curve,plot_pred

def train_model(tr_set,dv_set,config,device):
    model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device
    """# **Start Training!**"""
    model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
    plot_learning_curve(model_loss_record, title='deep model')
    del model

def test_model(tr_set,dv_set,tt_set,config,device):
    model = NeuralNet(tr_set.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')	# Load your best model
    model.load_state_dict(ckpt)
    plot_pred(dv_set, model, device)  # Show prediction on the validation set

    def save_pred(preds, file):
        ###Save predictions to specified file###
        print('Saving results to {}'.format(file))
        with open(file, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'tested_positive'])
            for i, p in enumerate(preds):
                writer.writerow([i, p])

    preds = test(tt_set, model, device)	 # predict COVID-19 cases with your model
    save_pred(preds, 'pred.csv')		 # save prediction file to pred.csv

def get_device():
	''' Get device (if GPU is available, use GPU) '''
	return 'cuda' if torch.cuda.is_available() else 'cpu'


tr_path = 'covid.train.csv'	 # path to training data
tt_path = 'covid.test.csv'	 # path to testing data

if __name__ == '__main__':
    """# **Setup Hyper-parameters**
    `config` contains hyper-parameters for training and the path to save your model.
    """
    device = get_device()				  # get the current available device ('cpu' or 'cuda')
    print(device)
    os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
    target_only = True					  # TODO: Using 40 states & 2 tested_positive features

    # TODO: How to tune these hyper-parameters to improve your model's performance?
    config = {
        'n_epochs': 3000,				 # maximum number of epochs
        'batch_size': 270,				 # mini-batch size for dataloader
        'optimizer': 'SGD',				 # optimization algorithm (optimizer in torch.optim)
        'optim_hparas': {				 # hyper-parameters for the optimizer (depends on which optimizer you are using)
            'lr': 0.001,				 # learning rate of SGD
            'momentum': 0.9,				 # momentum for SGD
            'weight_decay' : 1e-5
        },
        'early_stop': 200,				 # early stopping epochs (the number epochs since your model's last improvement)
        'save_path': 'models/model.pth'	 # your model will be saved here
    }

    """# **Load data and model**"""
    tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
    dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
    tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)
    
    train_model(tr_set,dv_set,config,device)
    test_model(tr_set,dv_set,tt_set,config,device)