from typing import Any, List
from abc import ABC, abstractmethod
from scipy.io import loadmat
import pandas as pd
import numpy as np 
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

def csvloader(file_path: str, **kwargs):
    '''
    Loads csv data
    '''
    file_data = pd.read_csv(file_path, index_col=False, header = 0).dropna().bfill()
    num_col = file_data.shape[1]
    num_extra_col = num_col % 3
    cols_to_select = num_col - num_extra_col
    activity_data = file_data.iloc[2:, -3:].to_numpy(dtype=np.float32)
    return activity_data

def matloader(file_path: str, **kwargs):
    '''
    Loads MatLab files 
    '''
    key = kwargs.get('key',None)
    assert key in ['d_iner' , 'd_skel'] , f'Unsupported {key} for matlab file'
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {
    'csv' : csvloader, 
    'mat' : matloader
}

def avg_pool(sequence : np.array, window_size : int = 5, stride :int =1, 
             max_length : int = 512 , shape : int = None) -> np.ndarray:

    '''
    Executes average pooling to smoothen out the data

    '''
    shape = sequence.shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis = 0).transpose(0,2, 1)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    stride =  ((sequence.shape[2]//max_length)+1 if max_length < sequence.shape[2] else 1)
    sequence = F.avg_pool1d(sequence,kernel_size=window_size, stride=stride)
    sequence = sequence.squeeze(0).numpy().transpose(1,0)
    sequence = sequence.reshape(-1, *shape[1:])
    return sequence


def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, 
                       input_shape: np.array) -> np.ndarray:
    '''
    Pools and pads the sequence to uniform length

    Args:
        sequence : data 
        max_sequence_length(int) : the fixed length of data
        input_shape: shape of the data
    Return: 
        new_sequence: data after padding
    '''
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length = max_sequence_length, shape = input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def sliding_window(data : np.ndarray, clearing_time_index : int, max_time : int, 
                   sub_window_size : int, stride_size : int) -> np.ndarray:
    '''
    Sliding Window
    '''
    assert clearing_time_index >= sub_window_size - 1 , "Clearing value needs to be greater or equal to (window size - 1)"
    start = clearing_time_index - sub_window_size + 1 

    if max_time >= data.shape[0]-sub_window_size:
        max_time = max_time - sub_window_size + 1
        # 2510 // 100 - 1 25 #25999 1000 24000 = 24900

    sub_windows  = (
        start + 
        np.expand_dims(np.arange(sub_window_size), 0) + 
        np.expand_dims(np.arange(max_time, step = stride_size), 0).T
    )

    #labels = np.round(np.mean(labels[sub_windows], axis=1))
    return data[sub_windows]

def selective_sliding_window(data: np.ndarray, length: int , window_size: int , stride_size : int, height : int, distance : int) -> np.array: 
    sqrt_sum = np.sqrt(np.sum(data**2, axis = 1))
    peaks , _ = find_peaks(sqrt_sum,height=height, distance=distance )
    windows = []
    for peak in peaks:
        start = max(0, peak - window_size)
        end = min(len(data), start + window_size)
        # difference = length - (end-start)
        # if difference != 0 : 
        #     if start == 0 : 
        #         end = end + difference
        #     elif 
        if data[start:end].shape[0] < window_size:
            continue
        windows.append(data[start:end])


            
                     
    #windows = [data[max(0, peak - W): min(len(data), peak + W)] for peak in peaks]
    return windows





class Processor(ABC):
    '''
    Data Processor 
    '''
    def __init__(self, file_path:str, mode : str, max_length: str, label: int, **kwargs):
        assert mode in ['sliding_window', 'avg_pool'], f'Processing mode: {mode} is undefined'
        self.label  = label 
        self.mode = mode
        self.max_length = max_length
        self.data = []
        self.file_path = file_path
        self.input_shape = []
        self.kwargs = kwargs


    def set_input_shape(self, sequence: np.ndarray) -> List[int]:
        '''
        returns the shape of the inputj

        Args: 
            sequence(np.ndarray) : data sequence
        
        Out: 
            shape (list) : shape of the sequence
        '''
        self.input_shape =  sequence.shape


    def _import_loader(self, file_path:str) -> np.array :
        '''
        Reads file and loads data from
         
        '''

        file_type = file_path.split('.')[-1]

        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'

        return LOADER_MAP[file_type]
    
    def load_file(self, file_path: str):
        '''
        Loads file given file path
        '''
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        self.set_input_shape(data)
        return data

    def process(self, data):
        '''
        function implementation to process data
        '''

        if self.mode == 'avg_pool':
            data = pad_sequence_numpy(sequence=data, max_sequence_length=self.max_length,
                                      input_shape=self.input_shape)
        
        else: 
            if self.label == 1: 
                #phone height = 25, distance = 200
                
                data = selective_sliding_window(data, length = self.input_shape[0],
                                                window_size= self.max_length, stride_size=10, height=1.4, distance=50)
            else: 
                #phone height = 15, distance = 500
                data = selective_sliding_window(data, length = self.input_shape[0],
                                                window_size= self.max_length, stride_size=10, height=1.2, distance=100)
                # data = sliding_window(data=data, clearing_time_index=self.max_length-1, 
                #                   max_time=self.input_shape[0],
                #                    sub_window_size =self.max_length, stride_size=10)
        return data

            