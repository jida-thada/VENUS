import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F 
import torch.optim as optim
from collections import OrderedDict
from utils import *



class pp_backbone(nn.Module):

    def __init__(self, backbone,hidden_dim,t_hiddim,stu_target_tasks,t_names,t_tasks,teachers,layername_list,layername_stu,t_modelnames):
        super(pp_backbone, self).__init__()
        self.stu = backbone 
        self.t_hiddim = t_hiddim
        self.stu_target_tasks = stu_target_tasks
        self.t_names = t_names
        self.t_tasks = t_tasks
        self.teachers = teachers
        self.layername_list = layername_list
        self.layername_stu = layername_stu
        self.t_modelnames = t_modelnames
        
        # Add adaptor to each teacher and each task: [A:[t1,t2],B:[t1,t2]]
        # t_hiddim = [[2,2],[2,None],[None,1]] - [[A-t1,A-t2],[B-t1,B-t2],[C-t1,C-t2]]
        self.adaptor_t = nn.ModuleList()  
        for tid, task_t_hidden in enumerate(t_hiddim):
            #print('task',tid)
            self.adaptor_taskj = nn.ModuleList()
            for t_dim in t_hiddim[tid]:
                if t_dim is not None:
                    self.adaptor_taskj.append(nn.Sequential(nn.Linear(hidden_dim,t_dim),
                                              nn.ReLU()))
            #print('self.adaptor_taskj',self.adaptor_taskj)
            self.adaptor_t.append(self.adaptor_taskj)
            #print('self.adaptor_t',self.adaptor_t)

    def forward(self, X):
        teacher_lastlayers = []
        for t_id, task in enumerate(self.stu_target_tasks):
            tea_FEATS = []
            for teacher_j, tname in enumerate(self.t_names):
                if task in self.t_tasks[teacher_j]:
                    teacher_model = self.teachers[teacher_j]
                    layername_list_tj = self.layername_list[teacher_j][task]
                    with torch.no_grad():
                        feat_extractor = RepExtractor(teacher_model,output_layers = layername_list_tj[0])
                        out, rep = feat_extractor(X)
                        if self.t_modelnames[teacher_j]=='alexnet' or self.t_modelnames[teacher_j]=='vgg':
                            feats = rep[layername_list_tj[0][0]] 
                        else: feats = rep[layername_list_tj[0][0]] 
                        tea_FEATS.append(feats)
            teacher_lastlayers.append(tea_FEATS)
        ft = teacher_lastlayers
        
        y_logit = self.stu(X); #print(y_logit)
        feat_extractor = RepExtractor(self.stu,output_layers = self.layername_stu)
        out, rep = feat_extractor(X)
        fs = rep[self.layername_stu[0]]; #print(fs.size())
        
        ft_ = []
        for task_id in range(len(self.t_hiddim)): 
            ft_j = []
            for rep_teacher_id, rep_teacher in enumerate(self.adaptor_t[task_id]):
                ft_j.append(rep_teacher[0](fs))
            ft_.append(ft_j)
        return y_logit, fs, (ft_, ft)




class MTL(nn.Module):
    
    def __init__(self, net,num_task,model_name):
        super().__init__()
        self.net = net
        self.num_task = num_task
        if model_name == 'resnet': 
            self.n_features = self.net.fc.out_features
        elif model_name == 'alexnet': 
            self.n_features = self.net.classifier[6].out_features
        elif model_name == 'vgg':
            self.n_features = self.net.classifier[6].out_features
        elif model_name == 'squeezenet':
            self.n_features = self.net.classifier[1].out_features
        elif model_name == 'densenet':
            self.n_features = self.net.classifier.out_features
        else: self.n_features = self.net.fc.out_features
        
        self.head_layers = nn.ModuleList()
        final_feat=16
        for tid in range(self.num_task):
            self.head_layers.append(nn.Sequential(
                OrderedDict([('linear', nn.Linear(self.n_features,final_feat)),
                             ('relu1', nn.ReLU()),
                             ('final', nn.Linear(final_feat, 1))])))
        
        
    def forward(self, x):
        for tid in range(self.num_task):
            if tid==0: self.logits = self.head_layers[tid](self.net(x))
            else: self.logits = torch.cat((self.logits,self.head_layers[tid](self.net(x))), 1)
        return self.logits
                                          
        
