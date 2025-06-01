import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from methods.tool_func import *


def preprocessing(x_fea):
  # x_fea: [B, 197, 384] --> x_cls_fea [B, 1, 384], x_patch_fea [B, 384, 14, 14]
  B, num, dim = x_fea.size()[0], x_fea.size()[1], x_fea.size()[2]
  x_cls_fea = x_fea[:, :1, :]
  x_patch_fea = x_fea[:,1:, :]
  x_patch_fea = x_patch_fea.contiguous().view(B,dim,num-1).view(B, dim, 14, 14)
  return x_cls_fea, x_patch_fea

def postprocessing(x_cls_fea, x_patch_fea):
  # x_cls_fea [B, 1, 384], x_patch_fea [B, 384, 14, 14] --> x_fea: [B, 197, 384] 
  B, num, dim = x_patch_fea.size()[0], x_patch_fea.size()[2]*x_patch_fea.size()[3]+1, x_patch_fea.size()[1]
  x_patch_fea = x_patch_fea.contiguous().view(B,dim,num-1).view(B,num-1,dim)
  x_fea = torch.cat((x_cls_fea, x_patch_fea), 1)
  return x_fea

def changeNewAdvStyle_ViT(vit_fea, new_styleAug_mean, new_styleAug_std, p_thred):
    if(new_styleAug_mean=='None'):
      return vit_fea

    #final
    p = np.random.uniform()
    if( p < p_thred):
      return vit_fea

    cls_fea, input_fea = preprocessing(vit_fea)
    feat_size = input_fea.size()
    ori_style_mean, ori_style_std = calc_mean_std(input_fea)
    #print('ori mean:', ori_style_mean.mean(), 'ori std:',  ori_style_std.mean())
    #print('adv mean:', new_styleAug_mean.mean(), 'adv std:', new_styleAug_std.mean())
    #print('mean diff:', new_styleAug_mean.mean() - ori_style_mean.mean(), 'std diff:', new_styleAug_std.mean() - ori_style_std.mean())
    normalized_fea = (input_fea - ori_style_mean.expand(feat_size)) / ori_style_std.expand(feat_size)
    styleAug_fea  = normalized_fea * new_styleAug_std.expand(feat_size) + new_styleAug_mean.expand(feat_size)
    styleAug_fea_vit = postprocessing(cls_fea, styleAug_fea)
    return styleAug_fea_vit

class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        # backbone
        self.feature = backbone
        final_feat_dim = 384
        self.classifier = nn.Linear(final_feat_dim, 64)

        self.loss_fn = nn.CrossEntropyLoss()

    def cos_classifier(self, w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim()-1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim()-1, eps=1e-12)

        cls_scores = f @ w.transpose(1, 2) # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores

    def forward(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        num_classes = supp_y.max() + 1 # NOTE: assume B==1
        B, nSupp, C, H, W = supp_x.shape
        supp_f = self.feature.forward(supp_x.contiguous().view(-1, C, H, W))
        supp_f = supp_f.view(B, nSupp, -1)
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp

        # B, nC, nSupp x B, nSupp, d = B, nC, d
        prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0 if some classes got 0 images

        feat = self.feature.forward(x.view(-1, C, H, W))
        feat = feat.view(B, x.shape[1], -1) # B, nQry, d

        logits = self.cos_classifier(prototypes, feat) # B, nQry, nC
        return logits

    def set_statues_of_modules(self, flag):
      if(flag=='eval'):
        self.feature.eval()
        self.classifier.eval()
        #self.scale_cls.eval()
        #self.bias.eval()
      elif(flag=='train'):
        self.feature.train()
        self.classifier.train()
        #self.scale_cls.train()
        #self.bias.train()
      return


    def forward_protonet(self, episode_f,supp_y, B, nSupp, nQuery, num_classes):
        #print('episode_f:', episode_f.size())
        episode_f = episode_f.view(num_classes, nSupp + nQuery, -1)
        #print('episode_f:', episode_f.size())
        fea_dim = episode_f.size()[-1]
        supp_f = episode_f[:, :nSupp, :].contiguous().view(-1, fea_dim).unsqueeze(0)
        query_f = episode_f[:, nSupp:, :].contiguous().view(-1, fea_dim).unsqueeze(0)
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp
        # B, nC, nSupp x B, nSupp, d = B, nC, d
        prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0 if some classes got 0 images
        logits = self.cos_classifier(prototypes, query_f) # B, nQry, nC
        return logits

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
        x_ori_block1 = self.feature.forward_block1(x_ori)
        x_ori_block1_cls, x_ori_block1_P = preprocessing(x_ori_block1)
        feat_size_block1 = x_ori_block1_P.size()
        #print('x_ori_block1:', x_ori_block1.size(), x_ori_block1_P.size())
        ori_style_mean_block1, ori_style_std_block1 = calc_mean_std(x_ori_block1_P)
        # set them as learnable parameters
        ori_style_mean_block1  = torch.nn.Parameter(ori_style_mean_block1)
        ori_style_std_block1 = torch.nn.Parameter(ori_style_std_block1)
        ori_style_mean_block1.requires_grad_()
        ori_style_std_block1.requires_grad_()
        # contain ori_style_mean_block1 in the graph 
        x_normalized_block1 = (x_ori_block1_P - ori_style_mean_block1.detach().expand(feat_size_block1)) / ori_style_std_block1.detach().expand(feat_size_block1)
        x_ori_block1_P = x_normalized_block1 * ori_style_std_block1.expand(feat_size_block1) + ori_style_mean_block1.expand(feat_size_block1)
        x_ori_block1 = postprocessing(x_ori_block1_cls, x_ori_block1_P)
        #print('x_ori_block1:', x_ori_block1.size())

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
        x_ori_block1 = self.feature.forward_block1(x_ori)
        # update adv_block1
        x_adv_block1 = changeNewAdvStyle_ViT(x_ori_block1, adv_style_mean_block1, adv_style_std_block1, p_thred=0)
        # forward block2
        x_ori_block2 = self.feature.forward_block2(x_adv_block1)
        # calculate mean and std
        x_ori_block2_cls , x_ori_block2_P = preprocessing(x_ori_block2)
        feat_size_block2 = x_ori_block2_P.size()
        ori_style_mean_block2, ori_style_std_block2 = calc_mean_std(x_ori_block2_P)
        # set them as learnable parameters
        ori_style_mean_block2  = torch.nn.Parameter(ori_style_mean_block2)
        ori_style_std_block2 = torch.nn.Parameter(ori_style_std_block2)
        ori_style_mean_block2.requires_grad_()
        ori_style_std_block2.requires_grad_()
        # contain ori_style_mean_block1 in the graph 
        x_normalized_block2 = (x_ori_block2_P - ori_style_mean_block2.detach().expand(feat_size_block2)) / ori_style_std_block2.detach().expand(feat_size_block2)
        x_ori_block2_P = x_normalized_block2 * ori_style_std_block2.expand(feat_size_block2) + ori_style_mean_block2.expand(feat_size_block2)
        x_ori_block2 = postprocessing(x_ori_block2_cls, x_ori_block2_P)
        # pass the rest model
        x_ori_block3 = self.feature.forward_block3(x_ori_block2)
        x_ori_block4 = self.feature.forward_block4(x_ori_block3)
        x_ori_fea = self.feature.forward_rest(x_ori_block4)
        x_ori_output = self.classifier.forward(x_ori_fea)
        # calculate initial pred, loss and acc
        ori_pred = x_ori_output.max(1, keepdim=True)[1]
        ori_loss = self.loss_fn(x_ori_output, y_ori)
        ori_acc = (ori_pred == y_ori).type(torch.float).sum().item() / y_ori.size()[0]
        #print('ori_pred:', ori_pred, 'ori_loss:', ori_loss, 'ori_acc:', ori_acc)
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
        #print('adv_style_mean_block2:', adv_style_mean_block2.size(), 'adv_style_std_block2:', adv_style_std_block2.size()) 

      # add zero_grad
      self.feature.zero_grad()
      self.classifier.zero_grad()

      if('3' in blocklist and epsilon_list[2] != 0):
        x_ori_block1 = self.feature.forward_block1(x_ori)
        x_adv_block1 = changeNewAdvStyle_ViT(x_ori_block1, adv_style_mean_block1, adv_style_std_block1, p_thred=0)
        x_ori_block2 = self.feature.forward_block2(x_adv_block1)
        x_adv_block2 = changeNewAdvStyle_ViT(x_ori_block2, adv_style_mean_block2, adv_style_std_block2, p_thred=0)
        x_ori_block3 = self.feature.forward_block3(x_adv_block2)
        x_ori_block3_cls, x_ori_block3_P = preprocessing(x_ori_block3)
        # calculate mean and std
        feat_size_block3 = x_ori_block3_P.size()
        ori_style_mean_block3, ori_style_std_block3 = calc_mean_std(x_ori_block3_P)
        # set them as learnable parameters
        ori_style_mean_block3  = torch.nn.Parameter(ori_style_mean_block3)
        ori_style_std_block3 = torch.nn.Parameter(ori_style_std_block3)
        ori_style_mean_block3.requires_grad_()
        ori_style_std_block3.requires_grad_()
        # contain ori_style_mean_block3 in the graph 
        x_normalized_block3 = (x_ori_block3_P - ori_style_mean_block3.detach().expand(feat_size_block3)) / ori_style_std_block3.detach().expand(feat_size_block3)
        x_ori_block3_P = x_normalized_block3 * ori_style_std_block3.expand(feat_size_block3) + ori_style_mean_block3.expand(feat_size_block3)
        x_ori_block3 = postprocessing(x_ori_block3_cls, x_ori_block3_P)
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




    def set_forward_loss_StyAdv(self, SupportTensor,QueryTensor,SupportLabel, QueryLabel, GlobalID_S,GlobalID_Q, epsilon_list):
        ##################################################################
        '''
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]

        # to tacke the input data
        x_ori: [5, 21, 3, 224, 224], global_y: [5, 21]
        '''
        # to resize as x_ori: torch.Size([5, 21, 3, 224, 224]) global_y: torch.Size([5, 21])
        B = SupportTensor.size()[0]
        num_classes = SupportLabel.max() + 1 # NOTE: assume B==1 
        SupportTensor = SupportTensor.squeeze().view(num_classes, -1, 3, 224, 224)
        QueryTensor = QueryTensor.squeeze().view(num_classes, -1, 3, 224, 224)
        nSupp = SupportTensor.size()[1]
        nQuery = QueryTensor.size()[1]
        
        x_ori = torch.cat((SupportTensor, QueryTensor), dim=1)
        global_y = torch.cat((GlobalID_S.view(num_classes, nSupp), GlobalID_Q.view(num_classes, nQuery)), dim=1)
        #print('x_ori:', x_ori.size(), 'global_y:', global_y.size())
        ##################################################################

        # 0. first cp x_adv from x_ori
        x_adv = x_ori

        # 1. styleAdv
        self.set_statues_of_modules('eval')
        adv_style_mean_block1, adv_style_std_block1, adv_style_mean_block2, adv_style_std_block2, adv_style_mean_block3, adv_style_std_block3 = self.adversarial_attack_Incre(x_ori, global_y, epsilon_list)
        self.feature.zero_grad()
        self.classifier.zero_grad()
           
        # 2. forward and get loss
        self.set_statues_of_modules('train')
        x_ori = x_ori.cuda()
        x_size = x_ori.size()
        x_ori = x_ori.view(num_classes*(nSupp+nQuery), 3, 224, 224)
        global_y = global_y.view(num_classes*(nSupp+nQuery)).cuda()
        x_ori_block1 = self.feature.forward_block1(x_ori)
        x_ori_block2 = self.feature.forward_block2(x_ori_block1)
        x_ori_block3 = self.feature.forward_block3(x_ori_block2)
        x_ori_block4 = self.feature.forward_block4(x_ori_block3)
        x_ori_fea = self.feature.forward_rest(x_ori_block4)

        # 3. ori cls global loss    
        scores_cls_ori = self.classifier.forward(x_ori_fea)
        loss_cls_ori = self.loss_fn(scores_cls_ori, global_y)

        # 4. ori FSL scores and losses
        scores_fsl_ori = self.forward_protonet(x_ori_fea, SupportLabel,B, nSupp, nQuery, num_classes)
        scores_fsl_ori = scores_fsl_ori.view(num_classes*nQuery, -1)
        QueryLabel = QueryLabel.view(-1)
        loss_fsl_ori = self.loss_fn(scores_fsl_ori, QueryLabel)
      
        # 5. forward StyleAdv
        x_adv = x_adv.cuda()
        x_adv = x_adv.view(x_size[0]*x_size[1], x_size[2], x_size[3], x_size[4])
        x_adv_block1 = self.feature.forward_block1(x_adv)
        x_adv_block1_newStyle = changeNewAdvStyle_ViT(x_adv_block1, adv_style_mean_block1, adv_style_std_block1, p_thred = P_THRED)
        x_adv_block2 = self.feature.forward_block2(x_adv_block1_newStyle)
        x_adv_block2_newStyle = changeNewAdvStyle_ViT(x_adv_block2, adv_style_mean_block2, adv_style_std_block2, p_thred = P_THRED)
        x_adv_block3 = self.feature.forward_block3(x_adv_block2_newStyle)
        x_adv_block3_newStyle = changeNewAdvStyle_ViT(x_adv_block3, adv_style_mean_block3, adv_style_std_block3, p_thred = P_THRED)
        x_adv_block4 = self.feature.forward_block4(x_adv_block3_newStyle)
        x_adv_fea = self.feature.forward_rest(x_adv_block4)

        # 6. adv cls gloabl loss
        scores_cls_adv = self.classifier.forward(x_adv_fea)
        loss_cls_adv = self.loss_fn(scores_cls_adv, global_y)
  
        # 7. adv FSL scores and losses
        scores_fsl_adv = self.forward_protonet(x_adv_fea, SupportLabel,B, nSupp, nQuery, num_classes)
        scores_fsl_adv = scores_fsl_adv.view(num_classes*nQuery, -1)
        loss_fsl_adv = self.loss_fn(scores_fsl_adv, QueryLabel)
             
        return scores_fsl_ori, loss_fsl_ori, scores_cls_ori, loss_cls_ori, scores_fsl_adv, loss_fsl_adv, scores_cls_adv, loss_cls_adv


