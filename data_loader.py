# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

"""# **Preprocess**
We have three kinds of datasets:
* `train`: for training
* `dev`: for validation
* `test`: for testing (w/o target value)

## **Dataset**

The `COVID19Dataset` below does:
* read `.csv` files
* extract features
* split `covid.train.csv` into train/dev sets
* normalize features

Finishing `TODO` below might make you pass medium baseline.
"""
class COVID19Dataset(Dataset):
	''' Dataset for loading and preprocessing the COVID19 dataset '''
	def __init__(self,
				 path,
				 mode='train',
				 target_only=False):
		self.mode = mode

		# Read data into numpy arrays
		with open(path, 'r') as fp:
			data = list(csv.reader(fp))
			data = np.array(data[1:])[:, 1:].astype(float)

		if not target_only:
			feats = list(range(93))
		else:
			# TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
			pass

		if mode == 'test':
			# Testing data
			# data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
			data = data[:, feats]
			self.data = torch.FloatTensor(data)
		else:
			# Training data (train/dev sets)
			# data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
			target = data[:, -1]
			data = data[:, feats]
			
			# Splitting training data into train & dev sets
			if mode == 'train':
				indices = [i for i in range(len(data)) if i % 10 != 0]
			elif mode == 'dev':
				indices = [i for i in range(len(data)) if i % 10 == 0]
			
			# Convert data into PyTorch tensors
			self.data = torch.FloatTensor(data[indices])
			self.target = torch.FloatTensor(target[indices])

		# Normalize features (you may remove this part to see what will happen)
		self.data[:, 40:] = \
			(self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
			/ self.data[:, 40:].std(dim=0, keepdim=True)
		
		self.dim = self.data.shape[1]
		print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
			  .format(mode, len(self.data), self.dim))

	def __getitem__(self, index):
		# Returns one sample at a time
		if self.mode in ['train', 'dev']:
			# For training
			return self.data[index], self.target[index]
		else:
			# For testing (no target)
			return self.data[index]

	def __len__(self):
		# Returns the size of the dataset
		return len(self.data)

"""## **DataLoader**
A `DataLoader` loads data from a given `Dataset` into batches.
"""
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
	''' Generates a dataset, then is put into a dataloader. '''
	dataset = COVID19Dataset(path, mode=mode, target_only=target_only)	# Construct dataset
	dataloader = DataLoader(
		dataset, batch_size,
		shuffle=(mode == 'train'), drop_last=False,
		num_workers=n_jobs, pin_memory=True)							# Construct dataloader
	return dataloader