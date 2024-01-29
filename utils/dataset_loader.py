import scipy.io as sc
import numpy as np
from tqdm import tqdm
import os
#import cv2
from einops import rearrange, repeat
import torch
from torch import nn, einsum
import torch.utils.data as data
#from train_test_split_on_subjects import *
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms




def average_frames(frames):
    sum = np.zeros(frames[0].shape)
    for ele in frames:
        sum += ele
    mean = sum/len(frames)

    return mean

def sampling_frames(record, interval, length):
    print(record.shape)
    sub_frames = [record[0]]
    sampled_frames = []
    for i in range(1, len(record)):
        if i % interval == 0:

            sampled_frames.append(average_frames(sub_frames))
            sub_frames = []
        sub_frames.append(record[i])

    if len(sub_frames) > 0:
        sampled_frames.append(average_frames(sub_frames))

    sampled_length = len(sampled_frames)
    if sampled_length < length:
        padding = sampled_frames[-1]
        for i in range(length-sampled_length):
            sampled_frames.append(padding)

    return np.asarray(sampled_frames[0:length])

def sampling_overlapped_frames(record, interval, length, overlap):
    sub_frames = [record[0]]
    sampled_frames = []
    for i in range(1, len(record)):
        if i % interval == 0:

            sampled_frames.append(average_frames(sub_frames))
            sub_frames = sub_frames[-overlap:]
        sub_frames.append(record[i])

    if len(sub_frames) > 0:
        sampled_frames.append(average_frames(sub_frames))

    sampled_length = len(sampled_frames)
    if sampled_length < length:
        padding = sampled_frames[-1]
        for i in range(length-sampled_length):
            sampled_frames.append(padding)

    return np.asarray(sampled_frames[0:length])


def load_inertial_data(root):
    #root = root + '' + 'Inertial'
    file_list = os.listdir(root)
    data = {}

    # min 107 frames
    # max 326 frames
    # average 180 frames
#     min = 9999
#     max = -9999
#     max_name = ""
#     min_name = ""
    for file in tqdm(file_list):

        source_data = sc.loadmat(root+'/'+file)
        # label: ActionNum_SubjectNum_Time
        index = "_".join(file.split('_')[0:3])
        # 6 dimensionts
        # two parts: acceleration and gyroscope, 3 axes each
        record = source_data['d_iner']
        # record dim: [num_frame, feature_dim(6)]

        record = sampling_frames(record, 5, 64)

        data[index] = record

    return data

def load_skeleton_data(root):
    root = root + '' + 'Skeleton'
    file_list = os.listdir(root)
    data = {}

    # frequency: 30 FPS
    # min 41 frames
    # max 125 frames
    # average 67 frames
#     min = 9999
#     max = -9999
#     max_name = ""
#     min_name = ""
    for file in tqdm(file_list):

        source_data = sc.loadmat(root+'/'+file)
        # label: ActionNum_SubjectNum_Time
        index = "_".join(file.split('_')[0:3])
        # 60 dimensionts
        # 20 parts: different skeletion joints, 3 axes each
        record = source_data['d_skel']


        # record shape [num_frame, feature_dim(60)]
        record = record[[15,14,13,12,7,6,5,4,3,0,1,2,8,9,10,11,16,17,18,19],:,:]
        record = rearrange(record, 'p a f -> f (p a)')

#         length = record.shape[0]
#         if min > length:
#             min = length
#             min_name = index
#         if max < length:
#             max = length
#             max_name = index
        record = sampling_frames(record, 2, 64)


        data[index] = record

    return data

def load_depth_data(root):
    root = root + '' + 'Depth'
    file_list = os.listdir(root)
    data = {}

    # frequency: 30 FPS
    # min 41 frames
    # max 125 frames
    # average 67 frames


    model = models.resnet50(pretrained=False)
    pretrained = torch.load('models/resnet50-0676ba61.pth')
    model.load_state_dict(torch.load('models/resnet50-0676ba61.pth'))
    for param in model.parameters():
        param.requires_grad = False
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to('cuda:0')
    result = 0



    transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.449], std=[0.226]),
])

    ymax=255
    ymin=0
    xmax=3975
    xmin=0

    with torch.no_grad():
        model.eval()
        for file in tqdm(file_list):
            #file = "a1_s5_t3_depth.mat"
            source_data = sc.loadmat(root+'/'+file)
            # label: ActionNum_SubjectNum_Time
            index = "_".join(file.split('_')[0:3])

            # frame size: 320 x 240
            # each frame has 1 channel
            record = source_data['d_depth']


            # record shape [num_frame, feature_dim(60)]
            record = rearrange(record, 'h w f -> f h w')
            record = sampling_frames(record, 2, 64)

    #         ma = np.max(record)
    #         if xmax < ma:
    #             xmax = ma

    #         mi = np.min(record)
    #         if xmin > mi:
    #             xmin = mi
            record=np.round((ymax-ymin)*(record-xmin)/(xmax-xmin)+ymin)
            frames = []
            for i in range(0,64):
                frame = transf(record[i])
                frame = np.asarray(frame)
                frames.append(frame)
            frames = np.asarray(frames)
            frames = rearrange(frames, "b 1 h w -> b h w")
            frames = repeat(frames, "b h w -> b d h w", d=3)
            frames = torch.from_numpy(frames).float().to('cuda:0')
            frames = model(frames)
            frames = rearrange(frames, "b d1 d2 d3 -> b (d1 d2 d3)")
            data[index] = frames.cpu()

    return data

def load_RGB_data(root):
    root = root + '' + 'RGB'
    file_list = os.listdir(root)
    data = {}

    result = 0

    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((240,320),interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



    model = models.resnet50(pretrained=False)
    pretrained = torch.load('models/resnet50-0676ba61.pth')
    model.load_state_dict(torch.load('models/resnet50-0676ba61.pth'))
    for param in model.parameters():
        param.requires_grad = False
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to('cuda:0')

    with torch.no_grad():
        model.eval()

        # frequency: 24 FPS
        # min 32 frames
        # max 96 frames
        # average 52 frames
        for file in tqdm(file_list):

            #source_data = cv2.VideoCapture(root+'/'+file)
            # label: ActionNum_SubjectNum_Time
            index = "_".join(file.split('_')[0:3])


            # frame size: 640 x 480
            # each frame has 3 channels
            success, image = source_data.read()
            image = transf(image)
            frames_counter = 0
            frames = [np.asarray(image)]
            while success:
                success,image = source_data.read()
                if success:
                    image = transf(image)
                    image = np.asarray(image)
                    frames.append(image)
            frames = np.asarray(frames)
            frames = sampling_frames(frames, 2, 48)

            frames = torch.from_numpy(frames).float().to('cuda:0')

            frames = model(frames)

            frames = rearrange(frames, "b d1 d2 d3 -> b (d1 d2 d3)")

            data[index] = frames.cpu()


    return data

label_dict = {'a1':0,   'a2':1,   'a3':2,
              'a4':3,   'a5':4,   'a6':5,
              'a7':6,   'a8':7,   'a9':8,
              'a10':9,  'a11':10, 'a12':11,
              'a13':12, 'a14':13, 'a15':14,
              'a16':15, 'a17':16, 'a18':17,
              'a19':18, 'a20':19, 'a21':20,
              'a22':21, 'a23':22, 'a24':23,
              'a25':24, 'a26':25, 'a27':26,
             }


def  construct_dataset(root='', indexes = None, prefix='', postfix='', mode='train'):



    labels = []
    rgb_features = []
    depth_features = []
    inertial_features = []
    skeleton_features = []


    if not indexes:
        files_dir = root + '' + 'RGB'
        file_list = os.listdir(files_dir)
        indexes = []
        for file in file_list:
            index = "_".join(file.split('_')[0:3])
            indexes.append(index)

        indexes = sorted(indexes)

    print('Creating datasets...')
    for index in tqdm(indexes):
        label = label_dict[index.split('_')[0]]
        labels.append(label)
        inertial_feature = torch.tensor(inertial_data[index], dtype=torch.float32, requires_grad=False)
        skeleton_feature = torch.tensor(skeleton_data[index], dtype=torch.float32, requires_grad=False)
        #depth_feature = torch.tensor(depth_data[index], dtype=torch.float32, requires_grad=False)
        #rgb_feature = torch.tensor(rgb_data[index], dtype=torch.float32, requires_grad=False)


        inertial_features.append(inertial_feature)
        skeleton_features.append(skeleton_feature)
        #depth_features.append(depth_feature)
        #rgb_features.append(rgb_feature)

    labels = torch.tensor(labels)
    inertial_features = torch.stack(inertial_features)
    skeleton_features = torch.stack(skeleton_features)
    #depth_features = torch.stack(depth_features)
    #rgb_features = torch.stack(rgb_features)

    print('Labels shape: {shape}'.format(shape=labels.shape))
    #print(rgb_features.shape)
    #print(depth_features.shape)
    print(inertial_features.shape)
    print(skeleton_features.shape)
    #print(labels.shape)

    inertial_dataset = torch.utils.data.TensorDataset(inertial_features, labels)
    torch.save(inertial_dataset, prefix+"inertial_Data"+postfix)
    print('Inertial dataset saved. The shape is {shape}...'.format(shape=inertial_features.shape))

    skeleton_dataset = torch.utils.data.TensorDataset(skeleton_features, labels)
    torch.save(skeleton_dataset, prefix+"skeleton_Data"+postfix)
    print('Skeleton dataset saved. The shape is {shape}...'.format(shape=skeleton_features.shape))

    # depth_dataset = torch.utils.data.TensorDataset(depth_features, labels)
    # torch.save(depth_dataset, prefix+"depth_Data_loso.pt")
    # print('Depth dataset saved. The shape is {shape}...'.format(shape=depth_features.shape))


    #rgb_dataset = torch.utils.data.TensorDataset(rgb_features, labels)
    #torch.save(rgb_dataset, prefix+"RGB_Data_loso.pt")
    #print('RGB dataset saved. The shape is {shape}...'.format(shape=rgb_features.shape))


if __name__ == "__main__":

    root = '/home/bgu9/Fall_Detection_KD_Multimodal/data/UTD_MAAD/train_inertial/'
    print('Loading inertial data from .mat files...')
    inertial_data = load_inertial_data(root)
    for values in inertial_data.values():
        print(values.shape)
    print('Loading skeleton data from .mat files...')
    skeleton_data = load_skeleton_data(root)
    print('Loading depth data from .mat files...')
    depth_data = load_depth_data(root)
    print('Loading RGB data from .mat files...')
    rgb_data = load_RGB_data(root)

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"


    # construct_dataset(root, train_50_1, '/home/jingcheng/source_code/extend/sample/si/train/large_train_1_','_Data_subjects.pt','train')
    # construct_dataset(root, test_50_1, '/home/jingcheng/source_code/extend/sample/si/test/large_test_1_','_Data_subjects.pt','test')
