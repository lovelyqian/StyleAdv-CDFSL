import torch
import torch.nn as nn
import numpy as np
import random

from methods.gnn import GNN_nl
from methods import backbone_multiblock
from methods.tool_func import *
from methods.meta_template_StyleAdv_RN_GNN import MetaTemplate


class StyleAdvGNN(MetaTemplate):
  maml=False
  def __init__(self, model_func,  n_way, n_support, tf_path=None):
    super(StyleAdvGNN, self).__init__(model_func, n_way, n_support, tf_path=tf_path)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml else nn.Sequential(backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
    self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)

    # for global classifier
    self.method = 'GnnNet'
    self.classifier = nn.Linear(self.feature.final_feat_dim, 64)

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way)

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    self.classifier.cuda()
    self.support_label = self.support_label.cuda()
    return self

  def set_forward(self,x,is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))
      z = z.view(self.n_way, -1, z.size(1))

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores = self.forward_gnn(z_stack)
    return scores



  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores


  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss


  def adversarial_attack_Incre(self, x_ori, y_ori, epsilon_list):
    x_ori = x_ori.cuda()
    y_ori = y_ori.cuda()
    x_size = x_ori.size()
    x_ori = x_ori.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    y_ori = y_ori.view(x_size[0]*x_size[1])

    # if not adv, set defalut = 'None'
    adv_style_mean_block1, adv_style_std_block1 = 'None', 'None'
    adv_style_mean_block2, adv_style_std_block2 = 'None', 'None'
    adv_style_mean_block3, adv_style_std_block3 = 'None', 'None'

    # forward and set the grad = True
    blocklist = 'block123'
    
    if('1' in blocklist and epsilon_list[0] != 0 ):
      # forward block1
      x_ori_block1 = self.feature.forward_block1(x_ori)
      feat_size_block1 = x_ori_block1.size()
      ori_style_mean_block1, ori_style_std_block1 = calc_mean_std(x_ori_block1)
      # set them as learnable parameters
      ori_style_mean_block1  = torch.nn.Parameter(ori_style_mean_block1)
      ori_style_std_block1 = torch.nn.Parameter(ori_style_std_block1)
      ori_style_mean_block1.requires_grad_()
      ori_style_std_block1.requires_grad_()
      # contain ori_style_mean_block1 in the graph 
      x_normalized_block1 = (x_ori_block1 - ori_style_mean_block1.detach().expand(feat_size_block1)) / ori_style_std_block1.detach().expand(feat_size_block1)
      x_ori_block1 = x_normalized_block1 * ori_style_std_block1.expand(feat_size_block1) + ori_style_mean_block1.expand(feat_size_block1)
      
      # pass the rest model
      x_ori_block2 = self.feature.forward_block2(x_ori_block1)
      x_ori_block3 = self.feature.forward_block3(x_ori_block2)
      x_ori_block4 = self.feature.forward_block4(x_ori_block3)
      x_ori_fea = self.feature.forward_rest(x_ori_block4)
      x_ori_output = self.classifier.forward(x_ori_fea)
    
      # calculate initial pred, loss and acc
      ori_pred = x_ori_output.max(1, keepdim=True)[1]
      ori_loss = self.loss_fn(x_ori_output, y_ori)
      ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]

      # zero all the existing gradients
      self.feature.zero_grad()
      self.classifier.zero_grad()
   
      # backward loss
      ori_loss.backward()

      # collect datagrad
      grad_ori_style_mean_block1 = ori_style_mean_block1.grad.detach()
      grad_ori_style_std_block1 = ori_style_std_block1.grad.detach()
    
      # fgsm style attack
      index = torch.randint(0, len(epsilon_list), (1, ))[0]
      epsilon = epsilon_list[index]

      adv_style_mean_block1 = fgsm_attack(ori_style_mean_block1, epsilon, grad_ori_style_mean_block1)
      adv_style_std_block1 = fgsm_attack(ori_style_std_block1, epsilon, grad_ori_style_std_block1)

    # add zero_grad
    self.feature.zero_grad()
    self.classifier.zero_grad()

    if('2' in blocklist and epsilon_list[1] != 0):
      # forward block1
      x_ori_block1 = self.feature.forward_block1(x_ori)
      # update adv_block1
      x_adv_block1 = changeNewAdvStyle(x_ori_block1, adv_style_mean_block1, adv_style_std_block1, p_thred=0)
      # forward block2
      x_ori_block2 = self.feature.forward_block2(x_adv_block1) 
      # calculate mean and std
      feat_size_block2 = x_ori_block2.size()
      ori_style_mean_block2, ori_style_std_block2 = calc_mean_std(x_ori_block2)
      # set them as learnable parameters
      ori_style_mean_block2  = torch.nn.Parameter(ori_style_mean_block2)
      ori_style_std_block2 = torch.nn.Parameter(ori_style_std_block2)
      ori_style_mean_block2.requires_grad_()
      ori_style_std_block2.requires_grad_()
      # contain ori_style_mean_block1 in the graph 
      x_normalized_block2 = (x_ori_block2 - ori_style_mean_block2.detach().expand(feat_size_block2)) / ori_style_std_block2.detach().expand(feat_size_block2)
      x_ori_block2 = x_normalized_block2 * ori_style_std_block2.expand(feat_size_block2) + ori_style_mean_block2.expand(feat_size_block2)
      # pass the rest model
      x_ori_block3 = self.feature.forward_block3(x_ori_block2)
      x_ori_block4 = self.feature.forward_block4(x_ori_block3)
      x_ori_fea = self.feature.forward_rest(x_ori_block4)
      x_ori_output = self.classifier.forward(x_ori_fea)
      # calculate initial pred, loss and acc
      ori_pred = x_ori_output.max(1, keepdim=True)[1]
      ori_loss = self.loss_fn(x_ori_output, y_ori)
      ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]
      # zero all the existing gradients
      self.feature.zero_grad()
      self.classifier.zero_grad()
      # backward loss
      ori_loss.backward()
      # collect datagrad
      grad_ori_style_mean_block2 = ori_style_mean_block2.grad.detach()
      grad_ori_style_std_block2 = ori_style_std_block2.grad.detach()
      # fgsm style attack
      index = torch.randint(0, len(epsilon_list), (1, ))[0]
      epsilon = epsilon_list[index]
      adv_style_mean_block2 = fgsm_attack(ori_style_mean_block2, epsilon, grad_ori_style_mean_block2)
      adv_style_std_block2 = fgsm_attack(ori_style_std_block2, epsilon, grad_ori_style_std_block2)

    # add zero_grad
    self.feature.zero_grad()
    self.classifier.zero_grad()

    if('3' in blocklist and epsilon_list[2] != 0):
      # forward block1, block2, block3
      x_ori_block1 = self.feature.forward_block1(x_ori)
      x_adv_block1 = changeNewAdvStyle(x_ori_block1, adv_style_mean_block1, adv_style_std_block1, p_thred=0)
      x_ori_block2 = self.feature.forward_block2(x_adv_block1)
      x_adv_block2 = changeNewAdvStyle(x_ori_block2, adv_style_mean_block2, adv_style_std_block2, p_thred=0)
      x_ori_block3 = self.feature.forward_block3(x_adv_block2)
      # calculate mean and std
      feat_size_block3 = x_ori_block3.size()
      ori_style_mean_block3, ori_style_std_block3 = calc_mean_std(x_ori_block3)
      # set them as learnable parameters
      ori_style_mean_block3  = torch.nn.Parameter(ori_style_mean_block3)
      ori_style_std_block3 = torch.nn.Parameter(ori_style_std_block3)
      ori_style_mean_block3.requires_grad_()
      ori_style_std_block3.requires_grad_()
      # contain ori_style_mean_block3 in the graph 
      x_normalized_block3 = (x_ori_block3 - ori_style_mean_block3.detach().expand(feat_size_block3)) / ori_style_std_block3.detach().expand(feat_size_block3)
      x_ori_block3 = x_normalized_block3 * ori_style_std_block3.expand(feat_size_block3) + ori_style_mean_block3.expand(feat_size_block3)
      # pass the rest model
      x_ori_block4 = self.feature.forward_block4(x_ori_block3)
      x_ori_fea = self.feature.forward_rest(x_ori_block4)
      x_ori_output = self.classifier.forward(x_ori_fea)
      # calculate initial pred, loss and acc
      ori_pred = x_ori_output.max(1, keepdim=True)[1]
      ori_loss = self.loss_fn(x_ori_output, y_ori)
      ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]
      # zero all the existing gradients
      self.feature.zero_grad()
      self.classifier.zero_grad()
      # backward loss
      ori_loss.backward()
      # collect datagrad
      grad_ori_style_mean_block3 = ori_style_mean_block3.grad.detach()
      grad_ori_style_std_block3 = ori_style_std_block3.grad.detach()
      # fgsm style attack
      index = torch.randint(0, len(epsilon_list), (1, ))[0]
      epsilon = epsilon_list[index]
      adv_style_mean_block3 = fgsm_attack(ori_style_mean_block3, epsilon, grad_ori_style_mean_block3)
      adv_style_std_block3 = fgsm_attack(ori_style_std_block3, epsilon, grad_ori_style_std_block3)

    return adv_style_mean_block1, adv_style_std_block1, adv_style_mean_block2, adv_style_std_block2, adv_style_mean_block3, adv_style_std_block3 
    
  
  def set_statues_of_modules(self, flag):
    if(flag=='eval'):
      self.feature.eval()
      self.fc.eval()
      self.gnn.eval()
      self.classifier.eval()
    elif(flag=='train'):
      self.feature.train()
      self.fc.train()
      self.gnn.train()
      self.classifier.train()
    return 
   

  def set_forward_loss_StyAdv(self, x_ori, global_y, epsilon_list):
    ##################################################################
    # 0. first cp x_adv from x_ori
    x_adv = x_ori

    ##################################################################
    # 1. styleAdv
    self.set_statues_of_modules('eval') 

    adv_style_mean_block1, adv_style_std_block1, adv_style_mean_block2, adv_style_std_block2, adv_style_mean_block3, adv_style_std_block3 = self.adversarial_attack_Incre(x_ori, global_y, epsilon_list)
 
    self.feature.zero_grad()
    self.fc.zero_grad()
    self.classifier.zero_grad()
    self.gnn.zero_grad()

    #################################################################
    # 2. forward and get loss
    self.set_statues_of_modules('train')

    # define y_query for FSL
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()

    # forward x_ori 
    x_ori = x_ori.cuda()
    x_size = x_ori.size()
    x_ori = x_ori.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    global_y = global_y.view(x_size[0]*x_size[1]).cuda()
    x_ori_block1 = self.feature.forward_block1(x_ori)
    x_ori_block2 = self.feature.forward_block2(x_ori_block1)
    x_ori_block3 = self.feature.forward_block3(x_ori_block2)
    x_ori_block4 = self.feature.forward_block4(x_ori_block3)
    x_ori_fea = self.feature.forward_rest(x_ori_block4)

    # ori cls global loss    
    scores_cls_ori = self.classifier.forward(x_ori_fea)
    loss_cls_ori = self.loss_fn(scores_cls_ori, global_y)
    acc_cls_ori = ( scores_cls_ori.max(1, keepdim=True)[1]  == global_y ).type(torch.float).sum().item() / global_y.size()[0]

    # ori FSL scores and losses
    x_ori_z = self.fc(x_ori_fea)
    x_ori_z = x_ori_z.view(self.n_way, -1, x_ori_z.size(1))
    x_ori_z_stack = [torch.cat([x_ori_z[:, :self.n_support], x_ori_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, x_ori_z.size(2)) for i in range(self.n_query)]
    assert(x_ori_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores_fsl_ori = self.forward_gnn(x_ori_z_stack)
    loss_fsl_ori = self.loss_fn(scores_fsl_ori, y_query)

    # forward x_adv
    x_adv = x_adv.cuda()
    x_adv = x_adv.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
    x_adv_block1 = self.feature.forward_block1(x_adv)

    x_adv_block1_newStyle = changeNewAdvStyle(x_adv_block1, adv_style_mean_block1, adv_style_std_block1, p_thred = P_THRED) 
    x_adv_block2 = self.feature.forward_block2(x_adv_block1_newStyle)
    x_adv_block2_newStyle = changeNewAdvStyle(x_adv_block2, adv_style_mean_block2, adv_style_std_block2, p_thred = P_THRED)
    x_adv_block3 = self.feature.forward_block3(x_adv_block2_newStyle)
    x_adv_block3_newStyle = changeNewAdvStyle(x_adv_block3, adv_style_mean_block3, adv_style_std_block3, p_thred = P_THRED)
    x_adv_block4 = self.feature.forward_block4(x_adv_block3_newStyle)
    x_adv_fea = self.feature.forward_rest(x_adv_block4)
   
    # adv cls gloabl loss
    scores_cls_adv = self.classifier.forward(x_adv_fea)
    loss_cls_adv = self.loss_fn(scores_cls_adv, global_y)
    acc_cls_adv = ( scores_cls_adv.max(1, keepdim=True)[1]  == global_y ).type(torch.float).sum().item() / global_y.size()[0]

    # adv FSL scores and losses
    x_adv_z = self.fc(x_adv_fea)
    x_adv_z = x_adv_z.view(self.n_way, -1, x_adv_z.size(1))
    x_adv_z_stack = [torch.cat([x_adv_z[:, :self.n_support], x_adv_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, x_adv_z.size(2)) for i in range(self.n_query)]
    assert(x_adv_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores_fsl_adv = self.forward_gnn(x_adv_z_stack)
    loss_fsl_adv = self.loss_fn(scores_fsl_adv, y_query)

    #print('scores_fsl_adv:', scores_fsl_adv.mean(), 'loss_fsl_adv:', loss_fsl_adv, 'scores_cls_adv:', scores_cls_adv.mean(), 'loss_cls_adv:', loss_cls_adv)
    return scores_fsl_ori, loss_fsl_ori, scores_cls_ori, loss_cls_ori, scores_fsl_adv, loss_fsl_adv, scores_cls_adv, loss_cls_adv
