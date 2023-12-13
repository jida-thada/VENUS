import numpy as np
#from flopth import flopth
import pickle
import random
from collections import Counter
#from sklearn.model_selection import StratifiedShuffleSplit
import collections
import itertools
from itertools import compress
import csv

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data

from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import glob
from collections import OrderedDict

import ssl
import math
from os.path import join
import os


import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = "serif"
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize':'x-large',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


def write_row(row, name, path="./", round=False):
    if round:
        row = [np.round(i, 2) for i in row]
    f = path + name + ".csv"
    with open(f, "a+") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(row)

def save_model(modeltosave, pathtosave):
    pickle.dump(modeltosave, open(pathtosave, 'wb'))


class RepExtractor(nn.Module):
    def __init__(self, submodule, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        self.selected_out = OrderedDict()
        self.model = submodule
        self.fhooks = []

        for i,l in enumerate(list(self.model.net._modules.keys())):
            if l in self.output_layers:
                self.fhooks.append(getattr(self.model.net,l).register_forward_hook(self.forward_hook(l)))    
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x): 
        out = self.model(x)
        return out, self.selected_out



    
def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
     """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def get_num_feat(model_name, model_ft=None, use_pretrained=False):
    if model_name == "resnet":
        """ Resnet18
        """
        if model_ft is None: model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        fc_out_feat = model_ft.fc.out_features
        fc_layername = 'fc'

    elif model_name == "alexnet":
        """ Alexnet
        """
        if model_ft is None: model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        fc_out_feat = model_ft.classifier[6].out_features
        fc_layername = 'classifier'

    elif model_name == "vgg":
        """ VGG11_bn
        """
        if model_ft is None: model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        fc_out_feat = model_ft.classifier[6].out_features
        fc_layername = 'classifier'

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        if model_ft is None: model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[1].in_features
        fc_out_feat = model_ft.classifier[1].out_features
        fc_layername = 'classifier'
        
    elif model_name == "densenet":
        """ Densenet
        """
        if model_ft is None: model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        fc_out_feat = model_ft.classifier.out_features
        fc_layername = 'classifier'
    else:
        num_ftrs = model_ft.fc.in_features
        fc_out_feat = model_ft.fc.out_features
        fc_layername = 'fc'
        
    return num_ftrs, fc_out_feat, fc_layername

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            


class data_prep(data.Dataset):
    
    def __init__(self, data_dir,img_transforms,
                 stu_tasks=None,
                 tr_stu_ratio= .67,te_stu_ratio = .33,
                 tr_stu_flag=False,te_stu_flag=False,
                 seed_stu=None,
                 path='./'):
        
        self.path = path
        self.stu_tasks = stu_tasks
        self.img_transforms = img_transforms
        self.seed = seed_stu
        self.seed_stu = seed_stu
        
        
        self.labels_id2list = {}
        self.labels_id2lab = {}
        self.labels_lab2id = {}
        self.groundtruth = {}
        lab_id = 0
        for c in self.stu_tasks:
            self.labels_id2list[lab_id]=list() 
            self.labels_id2lab[lab_id]=c
            self.labels_lab2id[c]=lab_id
            self.groundtruth[c]=list() 
            lab_id+=1
            
        # Separate data into each task
        data_id_for_each_task = {}
        data_filename_for_each_task = {}
        for t in self.stu_tasks:
            data_id_for_each_task[t]=[]
            data_filename_for_each_task[t]=[]
        file_list = glob.glob(os.path.join(data_dir, '*'))

        for img_id, img_file in enumerate(file_list):
            file_name = img_file[img_file.rfind('/')+1:-4]
            for t_id,t in enumerate(self.stu_tasks):
                if file_name[(2*t_id):(2*t_id)+1] == str(1): 
                    data_id_for_each_task[t].append(img_id)
                    data_filename_for_each_task[t].append(img_file)
        
        # Split data
        self.images_list = []
        self.ground_truth_list = []  
        self.images_name_list = {}
        for t_id,t in enumerate(self.stu_tasks):
            self.images_name_list[t]=list()
        for t_id,t in enumerate(self.stu_tasks):
            d = data_filename_for_each_task[t]
            count = len(d)
            file_index = list(range(count))
            random.seed(self.seed_stu)
            random.shuffle(file_index)
            new_file_list = []
            for file_id in file_index:
                new_file_list.append(d[file_id])
            n_tr_stu = int(np.floor(tr_stu_ratio*count))
            self.images_list_stu = new_file_list
            n_stu_data = len(self.images_list_stu)
            stu_file_index = list(range(n_stu_data))
            
            random.seed(self.seed_stu)
            random.shuffle(stu_file_index)
            stu_new_file_list = []
            for file_id in stu_file_index:
                stu_new_file_list.append(self.images_list_stu[file_id])
                
            if tr_stu_flag: 
                self.images_list_c = stu_new_file_list[:n_tr_stu]
            elif te_stu_flag:
                self.images_list_c = stu_new_file_list[n_tr_stu:]
            
            self.images_name_list[t] = self.images_list_c
            
            # Append data and groundtruth
            self.images_list_tmp = []
            for img_name in self.images_list_c:
                im=self.get_img(img_name)
                self.images_list_tmp.append(im)
                file_name = img_name[img_file.rfind('/')+1:-4]
                for t_id1,t1 in enumerate(self.stu_tasks):
                    self.groundtruth[t1].append(int(file_name[(2*t_id1):(2*t_id1)+1]))
            self.images_list.extend(self.images_list_tmp)
                

        self.task_mask = {}
        task_run = self.groundtruth.keys()
        
        for t_id,t in enumerate(task_run): 
            mask_taski = [0]*len(self.groundtruth[t])
            index_pos = [i for i, e in enumerate(self.groundtruth[t]) if e == 1]
            index_neg = [i for i, e in enumerate(self.groundtruth[t]) if e == 0]
            
            random.seed(self.seed_stu)
            random.shuffle(index_neg)
            n_pos_taski = len(index_pos)
            
            for i_mask, mask in enumerate(mask_taski):
                for i_pos in index_pos: mask_taski[i_pos] = 1
                for i_neg in index_neg[:n_pos_taski]: mask_taski[i_neg] = 1
                
            self.task_mask[t] = mask_taski

    def get_img(self,image_name):
        img = Image.open(image_name)
        t_img = self.img_transforms(img)
        return t_img
                
    def __len__(self):
         return len(self.images_list)
    
    def __getitem__(self, index):
        img = self.images_list[index]
        sample = {}
        sample['image'] = img
        for task_name in self.groundtruth.keys():
            sample[task_name]=self.groundtruth[task_name][index]
            sample['mask_task_'+task_name]=self.task_mask[task_name][index]
        return sample
    
    
