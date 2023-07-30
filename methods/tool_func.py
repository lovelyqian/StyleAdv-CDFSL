
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F

EPS=0.00001
#P_THRED = 0.2
P_THRED = 0.4
START_EPS = 16/255

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def fgsm_attack(init_input, epsilon, data_grad):
    # random start init_input
    init_input = init_input + torch.empty_like(init_input).uniform_(START_EPS, START_EPS)

    sign_data_grad = data_grad.sign()
    adv_input = init_input + epsilon*sign_data_grad
    return adv_input

def changeNewAdvStyle(input_fea, new_styleAug_mean, new_styleAug_std, p_thred):
    if(new_styleAug_mean=='None'):
        return input_fea
    
    p = np.random.uniform()
    if( p < p_thred):
        return input_fea

    feat_size = input_fea.size()
    ori_style_mean, ori_style_std = calc_mean_std(input_fea)
    normalized_fea = (input_fea - ori_style_mean.expand(feat_size)) / ori_style_std.expand(feat_size)
    styleAug_fea  = normalized_fea * new_styleAug_std.expand(feat_size) + new_styleAug_mean.expand(feat_size)
    return styleAug_fea

def consistency_loss(scoresM1, scoresM2, type='euclidean'):
    if(type=='euclidean'):
        avg_pro = (scoresM1 + scoresM2)/2.0
        matrix1 = torch.sqrt(torch.sum((scoresM1 - avg_pro)**2,dim=1))
        matrix2 = torch.sqrt(torch.sum((scoresM2 - avg_pro)**2,dim=1))
        dis1 = torch.mean(matrix1)
        dis2 = torch.mean(matrix2)
        dis = (dis1+dis2)/2.0
    elif(type=='KL1'):
        avg_pro = (scoresM1 + scoresM2)/2.0
        matrix1 = torch.sum( F.softmax(scoresM1,dim=-1) * (F.log_softmax(scoresM1, dim=-1) - F.log_softmax(avg_pro,dim=-1)), 1)
        matrix2 = torch.sum( F.softmax(scoresM2,dim=-1) * (F.log_softmax(scoresM2, dim=-1) - F.log_softmax(avg_pro,dim=-1)), 1)
        dis1 = torch.mean(matrix1)
        dis2 = torch.mean(matrix2)
        dis = (dis1+dis2)/2.0
    elif(type=='KL2'):
        matrix = torch.sum( F.softmax(scoresM2,dim=-1) * (F.log_softmax(scoresM2, dim=-1) - F.log_softmax(scoresM1,dim=-1)), 1)
        dis = torch.mean(matrix)
    elif(type=='KL3'):
        matrix = torch.sum( F.softmax(scoresM1,dim=-1) * (F.log_softmax(scoresM1, dim=-1) - F.log_softmax(scoresM2,dim=-1)), 1)
        dis = torch.mean(matrix)
    else:
        return
    return dis
