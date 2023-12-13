import matplotlib.pylab as pylab
import os
import numpy as np
import pickle
import random
from collections import Counter
#from sklearn.model_selection import StratifiedShuffleSplit
import collections
import itertools
from itertools import compress
import math
import csv
from sklearn import metrics
import argparse

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torchvision.transforms.functional
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import glob
from utils import *
from model import *

import matplotlib.pyplot as plt
plt.switch_backend('agg')

params = {'legend.fontsize': 'x-large',
                             'axes.labelsize': 'x-large',
                             'axes.titlesize': 'x-large',
                             'xtick.labelsize': 'x-large',
                             'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)


def prepare_data( dataname,stu_tasks,seed_stu,model_name='resnet'):
    data_dir = './data/'+dataname+'/'
    raw_file_list = glob.glob(os.path.join(data_dir, '*'))
    img_file = raw_file_list[0]
    img_name_jpg = img_file[img_file.rfind('/')+1:]
    num_task = img_name_jpg.count('_')

    if model_name == "resnet": input_size = 224
    elif model_name == "alexnet": input_size = 224
    elif model_name == "vgg": input_size = 224

    img_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    row=['--- Load TRAIN data']; write_row(row, 'Progress', path='./', round=False)
    train_dataset = data_prep(data_dir, img_transforms, 
                              stu_tasks, 
                              tr_stu_flag=True,
                              seed_stu=seed_stu)
    row=['--- Load TEST data']; write_row(row, 'Progress', path='./', round=False)
    test_dataset = data_prep(data_dir, 
                             img_transforms, 
                             stu_tasks, 
                             te_stu_flag=True,
                             seed_stu=seed_stu)
    return train_dataset,test_dataset


def load_teacher_model(tname):
    num_teachers = len(tname)
    teachers = []
    for ti in range(num_teachers):
        path_ti = './teachers/'+tname[ti]+'.sav'
        tmodel = pickle.load(open(path_ti, 'rb'))
        teachers.append(tmodel)
    return teachers


def extract_teacher_logits(task_id,task,inputs,t_names,t_tasks,teachers):
    logit_taski = torch.empty(0)
    for teacher_j, tname in enumerate(t_names):
        tj_task = list(t_tasks[teacher_j])
        if task in tj_task:
            index_task = tj_task.index(task)
            with torch.no_grad():
                logit_tj_taski = teachers[teacher_j](inputs)[:, index_task].unsqueeze(1)
                if logit_taski.size()[0] == 0:
                    logit_taski = logit_tj_taski
                else:
                    logit_taski = 0.5 * torch.add(logit_taski, logit_tj_taski)
    return logit_taski

        
def train_MTL_student(model,task_set,train_dataset,test_dataset,
                      device,stu_target_tasks, t_names, t_tasks, teachers, 
                      bs,lr,ep,
                      layername_list=None, layername_stu=None):

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = optim.SGD(params_to_update, lr=lr)  
    loss_fn = []
    for tid, task in enumerate(task_set):
        loss_fn.append(nn.BCEWithLogitsLoss(reduction='none'))
    criteria_loss_layer = nn.MSELoss()
    Sig = nn.Sigmoid()

    for epoch in range(ep):
        model.train()
        final_loss = 0
        for i, tr_data in enumerate(train_loader):
            inputs = tr_data["image"].to(device=device)
            
            for yid, c in enumerate(task_set):
                if yid == 0: 
                    y = extract_teacher_logits(yid,c,inputs,t_names,t_tasks,teachers).to(device=device)
                else:
                    y_meanlogit_taski = extract_teacher_logits(yid,c,inputs,t_names,t_tasks,teachers).to(device=device)
                    y = torch.cat((y, y_meanlogit_taski), -1)

            y = y.to(device=device)

            # --- Forward pass ---
            y_pred, fs, (ft_, ft) = model(inputs)
            
            # --- Backward and optimize ---
            optimizer.zero_grad()

            # --- Task loss
            loss_total = []
            for yid, c in enumerate(task_set):
                # --- With mask
                task_name = task_set[yid]
                mask_taski = tr_data['mask_task_'+task_name].to(device=device)
                mask_taski = torch.tensor(mask_taski)

                l = loss_fn[yid]
                loss_tmp = l(y_pred[:, yid], Sig(y[:, yid].float()/0.1))*mask_taski
                if torch.sum(mask_taski) == 0:
                    loss_total.append(torch.sum(loss_tmp))
                else:
                    loss_total.append(torch.sum(loss_tmp)/torch.sum(mask_taski))

            # --- Common feature loss
            loss_layer = 0.0
            for task_id, task in enumerate(task_set):
                index_dummy = 0
                for teacher_j, tname in enumerate(t_names):
                    if task in t_tasks[teacher_j]:
                        ft_i = ft[task_id][index_dummy]
                        stu_rep = ft_[task_id][index_dummy]
                        loss_taski = criteria_loss_layer(stu_rep, ft_i)
                        loss_layer += loss_taski
                        index_dummy += 1

            loss = np.sum(loss_total)+(loss_layer/bs)            
            loss.backward()
            optimizer.step()
            final_loss += loss.item()

        if epoch == 0 or (epoch+1) % (ep/10) == 0:
            row = ['ep', epoch+1, '/', ep, ': Loss ',np.round(np.mean(final_loss), 4)]
            write_row(row, 'Loss', path='./', round=False)
            # Eval
            acc_name = 'ep'+str(epoch)+' train'
            train_acc = evaluation(device, model, train_dataset, acc_name,bs)
            acc_name = 'ep'+str(epoch)+' test'
            test_acc = evaluation(device, model, test_dataset, acc_name,bs, test_flag=True, train_data=train_dataset)

    return model


def train_pp(stu_tasks,t_tasks,t_modelname,tnames,teachers,backbone,bs,lr,ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    task_set = stu_tasks
    num_task = len(task_set)
    
    layername_list = []
    for tid, t in enumerate(t_modelname):
        layername_ti = {}
        name_ti = t
        for taskid, task in enumerate(task_set):
            if task in t_tasks[tid]:
                fc_in_feat, fc_out_feat, fc_layername = get_num_feat(name_ti, teachers[tid].net)
                if task in layername_ti.keys():
                    layername_ti[task].append[fc_layername]
                else:
                    layername_ti[task] = [[fc_layername]]
        layername_list.append(layername_ti)

    t_hiddim = []  ##[A:[t1,t2],B:[t1,t2]]
    for taskid, task in enumerate(task_set):
        t_hiddim_ti = [] # [['resnet','feat-True','LR-0.01'],['alexnet','feat-True','LR-0.01']]
        for tid, t in enumerate(t_modelname):
            name_ti = t
            if task in t_tasks[tid]:
                fc_in_feat, fc_out_feat, fc_layername = get_num_feat(name_ti, teachers[tid].net)
            else:
                fc_out_feat = None
            t_hiddim_ti.append(fc_out_feat)
        t_hiddim.append(t_hiddim_ti)

    net, input_size = initialize_model(backbone, num_task)
    backbone = MTL(net, num_task, backbone)
    fc_in_feat, fc_out_feat, fc_layername = get_num_feat(backbone, backbone.net)
    layername_stu = [fc_layername]
    
    model = pp_backbone(backbone, fc_out_feat, t_hiddim, task_set, t_names,
                             t_tasks, teachers, layername_list, layername_stu, t_modelname)
    model = model.to(device=device)
    model = train_MTL_student(model, task_set, train_dataset, test_dataset, device,
                                task_set, t_names, t_tasks, teachers,
                                bs,lr,ep,layername_list, layername_stu)
    save_model(model, './output/') 

def evaluation(device, model, eval_dataset, acc_name, bs, csv_filename='Performance', test_flag=False, train_data=None):
    test_acc = {}
    num_correct = {}
    y_len = {}
    if test_flag:
        label_id2lab = list(eval_dataset.groundtruth.keys())
        train_task = list(train_data.groundtruth.keys())
    else:
        label_id2lab = list(eval_dataset.groundtruth.keys())
    for cid, c in enumerate(label_id2lab):
        task_name = label_id2lab[cid]
        test_acc[task_name] = 0
        num_correct[task_name] = 0
        y_len[task_name] = 0

    eval_dataloader = DataLoader(eval_dataset, batch_size=bs, shuffle=True, drop_last=False)
    sigmoid = nn.Sigmoid()
    threshold = torch.tensor([0.5]).to(device=device)
    if test_flag: run_task = train_task
    else: run_task = label_id2lab


    output_metric = {}
    output_metric_prob = {}
    target_metric = {}
    for i, eval_data in enumerate(eval_dataloader):
        inputs = eval_data["image"].to(device=device)
        batch = inputs.size()[0]
        for yid, c in enumerate(run_task):
            if yid == 0:
                y = eval_data[c].reshape(batch, 1)
            else:
                y_tmp = eval_data[c].reshape(batch, 1)
                y = torch.cat((y, y_tmp), -1)
        y = y.to(device=device)

        with torch.no_grad():
            y_pred, fs, (ft_, ft) = model(inputs)
            
        output = sigmoid(y_pred)
        pred = (output > threshold).float()*1

        for tid, task in enumerate(run_task):
            mask_taski = eval_data['mask_task_'+task].to(device=device)
            mask_taski = torch.tensor(mask_taski)
            pred_ti = pred[:, tid]
            pred_ti_prob = output[:, tid]
            y_ti = y[:, tid]

            # --- Add task_mask
            mask_list = mask_taski.tolist()
            mask1_index = [i for i, e in enumerate(mask_list) if e == 1]
            indices = torch.tensor(mask1_index).to(device=device)
            pred_ti_selected = torch.index_select(pred_ti, 0, indices)
            pred_ti_selected_prob = torch.index_select(pred_ti_prob, 0, indices)
            y_ti_selected = torch.index_select(y_ti, 0, indices)
            num_correct_ti = torch.sum((pred_ti_selected == y_ti_selected).float())
            num_correct[task] += num_correct_ti
            y_len[task] += len(y_ti_selected)

            
    for cid, c in enumerate(run_task):
        task_name = c
        num_correct_ti = num_correct[task_name]
        y_ti = y_len[task_name]
        acc = (num_correct_ti/y_ti).to('cpu')

        if test_flag:
            if task_name in train_task:
                row = [acc_name, task_name, np.round(acc.numpy(), 4)]
                write_row(row, csv_filename,path='./', round=False)
        else:
            row = [acc_name, task_name, np.round(acc.numpy(), 4)]
            write_row(row, csv_filename, path='./', round=False)
    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    
    add_arg('-backbone',    nargs='+',  type=str,   required=True,  help='name of backbone model eg, resnet ')
    add_arg('-lr',                      type=float, default=0.01,   help='learning rate')
    add_arg('-ep',                      type=int,   default=50,    help='total epochs for training')
    add_arg('-bs',                      type=int,   default=16,      help='batch size')
    add_arg('-seed_stu',                type=int,   default=0,      help='seed')
    add_arg('-tname',       nargs='+',  type=str,   required=True,  help=''' name of teacher models, e.g., 't0' 't1' ''')
    add_arg('-t_modelname', nargs='+',  type=str,   required=True,  help=''' model name of each teacher, e.g., 'resnet' 'vgg' ''')
    add_arg('-dataname',    nargs='+',  type=str,   required=True,  help='dataname, e.g., data1 ')
    add_arg('-t_tasks',     nargs='+',  type=str,   required=True,  help=''' tasks handled by each teacher, e.g., '0 1 2' '2 3 4' ''')
    
    
    args = parser.parse_args()
    backbone = args.backbone[0]
    dataname = args.dataname[0]
    t_tasks_input = args.t_tasks 
    seed_stu = args.seed_stu
    lr = args.lr
    ep = args.ep
    bs = args.bs
    tname = args.tname
    t_modelname = args.t_modelname
    
    t_tasks = []
    stu_tasks = []
    for i, task_ti in enumerate(t_tasks_input):
        t_ti_str = task_ti.split()
        t_ti = []
        for ti in t_ti_str:
            t_ti.extend(ti)
            if ti not in stu_tasks:
                stu_tasks.extend(ti)
        t_tasks.append(t_ti)
    
    print('t_tasks',t_tasks)
    print('stu_tasks',stu_tasks)
        
    commontask = list(set.intersection(*map(set, t_tasks)))
    num_commontask = len(commontask)
    train_dataset,test_dataset = prepare_data(dataname,stu_tasks,seed_stu,model_name=backbone)
    teachers = load_teacher_model(tname)
    train_pp(stu_tasks,t_tasks,t_modelname,tname,teachers,backbone,bs,lr,ep)
    print('--- DONE ---')
    
    #python main.py -backbone resnet -tname t0_densenet t1_resnet -t_modelname densenet resnet -dataname data1 -t_tasks '2 0' '2 3 1' -ep 1
    
