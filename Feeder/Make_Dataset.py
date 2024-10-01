import pandas as pd
import numpy as np
import math
import torch
import random
import torch.nn.functional as F
import scipy.stats as s
from einops import rearrange
#from tsaug import TimeWarp, Reverse, Drift, AddNoise


#################### MAIN #####################

#CREATE PYTORCH DATASET
'''
Input Args:
data = ncrc or ntu
num_frames = mocap and nturgb+d frame count!
acc_frames = frames from acc sensor per action
'''


class Utd_Dataset(torch.utils.data.Dataset):
    def __init__(self, npz_file):
        # Load data and labels from npz file
        dataset = np.load(npz_file)
        self.dataset = dataset['data']
        self.labels = dataset['labels']
        self.num_samples = self.dataset.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get the batch containing the requested index
        data = self.dataset[index, :, : , :]
        data = torch.tensor(data)
        label = self.labels[index]
        label = label - 1
        label = torch.tensor(label)
        label = label.long()
        return data, label

class Berkley_mhad(torch.utils.data.Dataset):
    def __init__(self, npz_file):
        # Load data and labels from npz file
        dataset = np.load(npz_file)
        self.dataset = dataset['data']
        self.labels = dataset['labels']
        self.num_samples = self.dataset.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get the batch containing the requested index
        data = self.dataset[index, :, :]
        data = torch.tensor(data)
        label = self.labels[index]
        label = label - 1
        label = torch.tensor(label)
        label = label.long()
        return data, label

class Bmhad_mm(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size, transform = None):
        # Load data and labels from npz file
        #dataset = np.load(npz_file)
        self.acc_data = dataset['acc_data']
        self.skl_data = dataset['skl_data']
        self.labels = dataset['labels']
        self.num_samples = self.acc_data.shape[0]
        self.acc_seq = self.acc_data.shape[1]
        # self.skl_joints = self.skl_data.shape[2]
        # self.skl_seq = self.skl_data.shape[1]
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get the batch containing the requested index
        data = dict()
        skl_data = torch.tensor(self.skl_data[index, :, :, :])
        #skl_data = skl_data.reshape((l, -1, 3))
        acc_data = torch.tensor(self.acc_data[index, : , :])
        data['skl_data'] = skl_data
        data['acc_data'] =  acc_data
        label = self.labels[index]
        label = torch.tensor(label)
        label = label.long()
        return data, label, index
    

class UTD_mm(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size, transform = None):
        # Load data and labels from npz file
        #dataset = np.load(npz_file)
        self.acc_data = dataset['acc_data']
        self.skl_data = dataset['skl_data']
        self.labels = dataset['labels']
        self.num_samples = self.acc_data.shape[0]
        self.acc_seq = self.acc_data.shape[1]
        self.skl_joints = self.skl_data.shape[2]
        self.skl_seq = self.skl_data.shape[1]
        self.skl_channels = self.skl_data.shape[3]
        self.channels = self.acc_data.shape[2]
        self.batch_size = batch_size
        self.transform = None
        self.crop_size = 64
        # self.transform =  (
        #      AddNoise(scale=(0.01, 0.05)) @ 0.5  # random quantize to 10-, 20-, or 30- level sets
        #     + Drift(max_drift=(0.1, 0.5)) @ 0.4  # with 80% probability, random drift the signal up to 10% - 50%
        #     + Reverse() @ 0.5 ) # with 50% probability, reverse the sequence 
    
    def random_crop(self,data : torch.Tensor) -> torch.Tensor:
        '''
        Function to add random cropping to the data
        Arg: 
            data : 
        Output: 
            crop_data: will return croped data
        '''
        length = data.shape[0]
        start_idx = np.random.randint(0, length-self.crop_size-1)
        return data[start_idx : start_idx+self.crop_size, :]

    # def augment(self, data): 
    #     data = torch.transpose(data, -2, -1)
    #     transformed_data = self.transform.augment(data.numpy())
    #     data = torch.tensor(np.transpose(transformed_data))
    #     return data 

    def cal_smv(self, sample : torch.Tensor) -> torch.Tensor:
        '''
        Function to calculate SMV
        '''
        mean = torch.mean(sample, dim = -2, keepdim=True)
        zero_mean = sample - mean
        sum_squared =  torch.sum(torch.square(zero_mean), dim=-1, keepdim=True) 
        smv= torch.sqrt(sum_squared)
        return smv
    
    def calculate_weight(self, data):
        """
        Calculate the magnitude (weight) of accelerometer data.

        Parameters:
        - data: A PyTorch tensor of shape (128, 3) where each row is [ax, ay, az].

        Returns:
        - A 1D PyTorch tensor of shape (128,) containing the magnitude for each row.
        """
        return torch.sqrt(torch.sum(data**2, dim=-1, keepdim=True))
    
    def calculate_pitch(self,data):
        """
        Calculate the pitch from accelerometer data.

        Parameters:
        - data: A PyTorch tensor of shape (128, 3) where each row is [ax, ay, az].

        Returns:
        - A 1D PyTorch tensor of shape (128,) containing the pitch angle for each row in radians.
        """
        ax = data[:, 0]
        ay = data[:, 1]
        az = data[:, 2]
        return torch.atan2(ay, torch.sqrt(ax**2 + az**2))
    

    def calculate_roll(self,data):
        """
        Calculate the roll from accelerometer data.

        Parameters:
        - data: A PyTorch tensor of shape (128, 3) where each row is [ax, ay, az].

        Returns:
        - A 1D PyTorch tensor of shape (128,) containing the roll angle for each row in radians.
        """
        ax = data[:, 0]
        ay = data[:, 1]
        az = data[:, 2]
        return torch.atan2(ax, torch.sqrt(ay**2 + az**2))




    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get the batch containing the requested index
        skl_data = torch.tensor(self.skl_data[index, :, :,:])
        acc_data = torch.tensor(self.acc_data[index, : , :])
        # data = torch.zeros(self.skl_seq, self.skl_joints+self.acc_seq, self.channels)
        # data[:, :self.skl_joints, :self.skl_channels] = skl_data
        # data[ 0, self.skl_joints:, :self.channels] = acc_data
        data = dict()


        if self.transform and np.random.random() > 0.6:
            T, N , C = skl_data.shape
            skl_data = rearrange(skl_data, 'T N C -> T (N C)')
            aug_skl = self.augment(skl_data)
            aug_skl = rearrange(aug_skl, 'T (N C) -> T N C', N = N , C = C)
            skl_data = aug_skl
            acc_data = self.augment(acc_data)

        #data['acc_data'] = self.random_crop(acc_data)
        #calculating smv 
        watch_smv = self.cal_smv(acc_data)
        phone_smv = self.cal_smv(acc_data[:,3:])
        #phone_smv = self.cal_smv(acc_data)
        weight = self.calculate_weight(acc_data)
        #roll = torch.unsqueeze(self.calculate_roll(acc_data), dim = 1)
        #pitch = torch.unsqueeze(self.calculate_pitch(acc_data),dim =1)
        acc_data = torch.cat(( watch_smv,acc_data), dim = -1)
        #data['acc_data'] = self.random_crop(acc_data)
        data['acc_data'] = acc_data
        data['skl_data'] = skl_data
        label = self.labels[index]
        #data, label = self.transform(data, label)
        label = torch.tensor(label)
        label = label.long()
        return data, label, index

class UTD_aus(torch.utils.data.Dataset):
    def __init__(self):
        acc_data = torch.load('/home/bgu9/DMFT_wrapup/si/train/large_train_1_inertial_Data_subjects.pt')
        skl_data = torch.load('/home/bgu9/DMFT_wrapup/si/train/large_train_1_skeleton_Data_subjects.pt')
        print(acc_data.shape)
        print(skl_data.shape)

def cal_smv(sample : torch.Tensor) -> torch.Tensor:
        '''
        Function to calculate SMV
        '''
        mean = torch.mean(sample, dim = -2, keepdim=True)
        zero_mean = sample - mean
        sum_squared =  torch.sum(torch.square(zero_mean), dim=-1, keepdim=True) 
        smv= torch.sqrt(sum_squared)
        processed_data = torch.cat((sample, smv), dim = -1)
        return processed_data



if __name__ == "__main__":
    data = torch.randn((8, 128, 3))
    smv = cal_smv(data)