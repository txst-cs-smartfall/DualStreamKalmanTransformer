'''
Dataset Builder
'''
import os
from typing import List, Dict, Tuple
import numpy as np
from numpy.linalg import norm
from dtaidistance import dtw
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

from utils.processor.base import Processor

def filter_data_by_ids(data : np.ndarray, ids : List[int]):
    '''
    Index the different modalities with only selected ids

    Arguements: 
        data : data dictionary with skeleton and inertial data
        skeleton_ids: skeleton data selected ids
        inertial_ids: inertial data selected ids
    Return : 
        changed data with selected ids
    '''
    return data[ids]




def filter_repeated_ids(path : List[Tuple[int, int]]) -> Tuple[set, set]:
    '''
    Filtering indices those match with mutliple other indices
    Arguements: 
        path : Tuple of indices defining the DTW path
    
    Return : 
        set of tuples containing the unique indices

    '''
    seen_first = set()
    seen_second = set()

    for (first , second) in path : 

        if first not in seen_first and second not in  seen_second: 
            seen_first.add(first)
            seen_second.add(second)
    
    return seen_first, seen_second

def align_sequence(data : Dict[str, np.ndarray], idx ) -> Dict[str, np.ndarray]: 
    '''
    Matching the skeleton and phone data using dynamic time warping 
    Args: 
        dataset: Dictionary containing skeleton and accelerometer data

    '''
    joint_id = 9
    #skeleton_before_dtw =  data['skeleton'][idx][:, (joint_id -1) * 3 : joint_id * 3 ]
    #seperating left wrist joint data
    dynamic_keys = [key for key in data.keys() if key != "skeleton"][0]
    skeleton_joint_data = data['skeleton'][idx][:, (joint_id -1) * 3 : joint_id * 3 ]
    inertial_data = data[dynamic_keys][idx]

   # calcuating frobenis norm of skeleton and intertial data 
    skeleton_frob_norm = norm(skeleton_joint_data, axis = 1)
    interial_frob_norm = norm(inertial_data, axis = 1)
    
    # calculating dtw of the two sequence
    path =  dtw.warping_path(
        skeleton_frob_norm.flatten(), 
        interial_frob_norm.flatten()
    )

    skeleton_idx , interial_ids = filter_repeated_ids(path)
    data['skeleton'][idx] = filter_data_by_ids(data['skeleton'][idx], list(skeleton_idx))
    data[dynamic_keys][idx]= filter_data_by_ids(data[dynamic_keys][idx],list(interial_ids))
    #skeleton_after_dtw = data['skeleton'][idx][:, (joint_id -1) * 3 : joint_id * 3 ]
    #plt.plot( np.arange(skeleton_before_dtw.shape[0]),skeleton_before_dtw[..., 0], '--r',
             #np.arange(skeleton_after_dtw.shape[0]), skeleton_after_dtw[..., 0], '--g')
    
    # plt.savefig(f'exps/comparision/comparision_before_after_dtw_{idx}.jpg')
    # plt.close()
    return data


def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    '''Function to fitter noise '''
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data, axis=0) 

class DatasetBuilder:
    '''
    Builds a numpy file for the data and labels and 

    Args: 
        Dataset: a dataset class containing all matched files
    '''
    def __init__(self , dataset: object, mode: str, max_length: int, task = 'fd', **kwargs) -> None:
        assert mode in ['avg_pool' , 'sliding_window'], f'Unsupported processing method {mode}'
        self.dataset = dataset
        self.data : Dict[str, List[np.array]] = {}
        self.processed_data : Dict[str, List[np.array]] = {'labels':[]}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task

    
    def make_dataset(self, subjects : List[int]): 
        '''
        Reads all the files and makes a numpy  array with all data
        '''
        self.data = {}
        self.processed_data : Dict[str, List[np.array]] = {'labels':[]}
        count = 0 
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:       
                if self.task == 'fd': 
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                #self.data['labels'] = self.data.get('labels',[])
                
                for modality, file_path in trial.files.items():
                    #here we need the processor class 
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys:
                        key = keys[modality.lower()]
                    processor = Processor(file_path, self.mode, self.max_length, key = key)
                    try: 
                        #unimodal_data = butterworth_filter(processor.process(), cutoff=1.0, fs=20)
                        unimodal_data = processor.load_file()
                        #print(f"Modality : { modality} , shape : {unimodal_data.shape}")
                        self.data[modality] = self.data.get(modality, [])
    
                        self.data[modality].append(unimodal_data)

                    except Exception as e : 
                        print(e)
                        # os.remove(file_path)
                try:
                    self.data = align_sequence(self.data, count)
                except:
                    continue
                
                for modality, file_path in trial.files.items():
   
                    processor = Processor(file_path, self.mode, self.max_length, key = key)
                    processor.set_input_shape(self.data[modality][count])
                    window_stack = processor.process(self.data[modality][count])
                    if window_stack.shape[0] != 0 :
                        self.processed_data[modality] = self.processed_data.get(modality, [])
                        self.processed_data[modality].append(window_stack)
                if processor.input_shape[0] >= self.max_length:
                    self.processed_data['labels'].append(np.repeat(label,window_stack.shape[0]))

                    #print(self.data['skeleton'][1].shape)
                count +=1
 
        for key in self.processed_data:
            
            self.processed_data[key] = np.concatenate(self.processed_data[key], axis=0)
        

    
    def normalization(self) -> np.ndarray:
        '''
        Function to normalize  the data
        '''

        for key ,value  in self.processed_data.items():        
            if key != 'labels':
                num_samples, length = value.shape[:2]
                norm_data = StandardScaler().fit_transform(value.reshape(num_samples*length, -1))
                self.processed_data[key] = norm_data.reshape(num_samples, length, -1)

        return self.processed_data
    
