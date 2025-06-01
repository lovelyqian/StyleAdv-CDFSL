import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import random
from methods.backbone import model_dict
from data.datamgr import SetDataManager
from options import parse_args
#from methods.matchingnet import MatchingNet
#from methods.relationnet import RelationNet
#from methods.protonet import ProtoNet
#from methods.gnnnet import GnnNet
#from methods.tpn import TPN
#from PSG import PseudoSampleGenerator
from utils.PSG import PseudoSampleGenerator

from data import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

#from cvpr2023_startup_20221026 import *
#from cvpr2023_load_models_20221102 import load_ViTsmall
from methods.load_ViT_models import load_ViTsmall
#from models.pmf_protonet import ProtoNet
#from methods.pmf_protonet import ProtoNet
from methods.protonet import ProtoNet

#PMF_metatrained = False
PMF_metatrained = True
FINAL_FEAT_DIM = 384
FINETUNE_ALL = True
#FINETUNE_ALL = False

#tune_lr = 0.01
#tune_lr = 0.001
#tune_lr = 0.0001
tune_lr = 5e-5

def load_model():
  vit_model = load_ViTsmall()
  model = ProtoNet(vit_model)

  if PMF_metatrained:
    #pmf_pretrained_ckp = 'outputs/20221103-styleAdv_metatrain_vit_protonet_trainEpoch20_exp0_lr0/checkpoint.pth'
    #pmf_pretrained_ckp = 'outputs/20221103-styleAdv_metatrain_vit_protonet_trainEpoch20_exp1_lr1/checkpoint.pth'
    #pmf_pretrained_ckp = 'outputs/20221103-styleAdv_metatrain_vit_protonet_trainEpoch20_exp2_lr2/checkpoint.pth'
    #pmf_pretrained_ckp = 'outputs/20221103-styleAdv_metatrain_vit_protonet_trainEpoch20_exp3_lr3/checkpoint.pth'

    # 1shot
    #pmf_pretrained_ckp = 'outputs/20221106-styleAdv_metatrain_vit_protonet_trainEpoch20_1shot_exp0_lr0_saveBestPth/checkpoint.pth'
    # pmf_pretrained_ckp = 'output/20221106-styleAdv_metatrain_vit_protonet_trainEpoch20_1shot_exp2_lr2_saveBestPth/checkpoint.pth'
    #pmf_pretrained_ckp = 'outputs/20221106-styleAdv_metatrain_vit_protonet_trainEpoch20_1shot_exp0_lr0_saveBestPth_PthreDot4/checkpoint.pth'
    #pmf_pretrained_ckp = 'outputs/20221106-styleAdv_metatrain_vit_protonet_trainEpoch20_1shot_exp2_lr2_saveBestPth_PthreDot4/checkpoint.pth'
 
    #pmf_pretrained_ckp = 'outputs/20221106-withoutstyleAdv_metatrain_vit_protonet_exp0_1shot/best.pth'
    
    # 1shot-update todo
    pmf_pretrained_ckp = 'output/20230723-test-ViT/best.pth'
    state_pmf = torch.load(pmf_pretrained_ckp)['model']
    
    #
    state_new = state_pmf
    state_keys = list(state_pmf.keys())
    for i, key in enumerate(state_keys):
      if 'feature.' in key:
        newkey = key.replace("feature.","backbone.")
        state_new[newkey] = state_pmf.pop(key)
      if 'classifier.' in key:
        state_new.pop(key)
      else:
        pass
    model.load_state_dict(state_new)
  model.train().cuda()
  return model


def set_forward_ViTProtonet(model, x):
        n_way = x.size()[0]
        n_query = 15
        n_support = x.size()[1] - n_query

        SupportTensor = x[:, :n_support, :, :, :]
        QueryTensor = x[:, n_support:, :, :, :]
        SupportLabel = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
        QueryLabel = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()

        SupportTensor = SupportTensor.contiguous().view(-1, n_way*n_support, 3, 224, 224)
        QueryTensor = QueryTensor.contiguous().view(-1, n_way*n_query, 3, 224, 224)
        SupportLabel = SupportLabel.contiguous().view(-1, n_way*n_support)
        QueryLabel = QueryLabel.contiguous().view(-1,  n_way*n_query)
        #print(SupportTensor.size(), SupportLabel.size(), QueryTensor.size())
        output = model(SupportTensor, SupportLabel, QueryTensor)
        output = output.view(n_way*n_query,n_way)
        return output

def finetune(novel_loader, n_pseudo=75, n_way=5, n_support=5):
    iter_num = len(novel_loader)
    acc_all = []

    #checkpoint_dir = '%s/checkpoints/%s/best_model.tar' % (params.save_dir, params.name)
    #checkpoint_dir = '%s/checkpoints/%s/best_model.tar' % (params.save_dir, params.resume_dir)
    #state = torch.load(checkpoint_dir)['state']
    for ti, (x, _) in enumerate(novel_loader):  # x:(5, 20, 3, 224, 224)
        '''
        # Model
        if params.method == 'MatchingNet':
            model = MatchingNet(model_dict[params.model], n_way=n_way, n_support=n_support).cuda()
        elif params.method == 'RelationNet':
            model = RelationNet(model_dict[params.model], n_way=n_way, n_support=n_support).cuda()
        elif params.method == 'ProtoNet':
            model = ProtoNet(model_dict[params.model], n_way=n_way, n_support=n_support).cuda()
        elif params.method == 'GNN':
            model = GnnNet(model_dict[params.model], n_way=n_way, n_support=n_support).cuda()
        elif params.method == 'TPN':
            model = TPN(model_dict[params.model], n_way=n_way, n_support=n_support).cuda()
        else:
            print("Please specify the method!")
            assert (False)
        # Update model
        if 'FWT' in params.name:
            model_params = model.state_dict()
            pretrained_dict = {k: v for k, v in state.items() if k in model_params}
            model_params.update(pretrained_dict)
            model.load_state_dict(model_params)
        else:
            model.load_state_dict(state, strict = False)
        '''
        model = load_model()
        x = x.cuda()
        # Finetune components initialization
        xs = x[:, :n_support].reshape(-1, *x.size()[2:])  # (25, 3, 224, 224)
        #print('xs:', xs.size())
        pseudo_q_genrator = PseudoSampleGenerator(n_way, n_support, n_pseudo)
        loss_fun = nn.CrossEntropyLoss().cuda()
        #opt = torch.optim.Adam(model.parameters())
        #opt = torch.optim.Adam(model.parameters(), lr=0.0005)  #lr version 2
        opt = torch.optim.SGD(model.parameters(), lr = tune_lr, momentum=0.9, weight_decay=0,) #pmf opt

        # Finetune process
        n_query = n_pseudo//n_way
        pseudo_set_y = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
        model.n_query = n_query
        model.train()
        for epoch in range(params.finetune_epoch):
            opt.zero_grad()
            pseudo_set = pseudo_q_genrator.generate(xs)  # (5, n_support+n_query, 3, 224, 224)
            #scores = model.set_forward(pseudo_set)  # (5*n_query, 5)
            scores = set_forward_ViTProtonet(model, pseudo_set)
            loss = loss_fun(scores, pseudo_set_y)
            loss.backward()
            opt.step()
            del pseudo_set, scores, loss
        torch.cuda.empty_cache()

        # Inference process
        n_query = x.size(1) - n_support
        model.n_query = n_query
        yq = np.repeat(range(n_way), n_query)
        with torch.no_grad():
            #scores = model.set_forward(x)  # (80, 5)
            scores = set_forward_ViTProtonet(model, x)
            _, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()  # (80, 1)
            top1_correct = np.sum(topk_ind[:,0]==yq)
            acc = top1_correct*100./(n_way*n_query)
            acc_all.append(acc)
        del scores, topk_labels
        torch.cuda.empty_cache()
        #print('Task %d : %4.2f%%'%(ti, acc))
        #print('Task %d : %4.2f%%, mean Acc: %4.2f'%(ti, acc, np.mean(np.array(acc_all))))
        if(ti%50==0):
          print('Task %d : %4.2f%%, mean Acc: %4.2f'%(ti, acc, np.mean(np.array(acc_all))))

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('Test Acc = %4.2f +- %4.2f%%'%(acc_mean, 1.96*acc_std/np.sqrt(iter_num)))

def run_single_testset(params):
    seed = 0
    #print("set seed = %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #np.random.seed(10)
    #params = parse_args('train')

    #params = parse_args()

    image_size = 224
    iter_num = 1000
    n_query = 15
    n_pseudo = 75
    #print('n_pseudo: ', n_pseudo)

    print('Loading target dataset!:', params.testset)
    if params.testset in ['cub', 'cars', 'places', 'plantae']:
      novel_file = os.path.join(params.data_dir, params.testset, 'novel.json')
      datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot, n_eposide=iter_num)
      novel_loader = datamgr.get_data_loader(novel_file, aug=False)
    
    else:
      few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
      if params.testset in ["ISIC"]:
        datamgr         = ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = n_query, **few_shot_params)
        novel_loader     = datamgr.get_data_loader(aug = False )

      elif params.testset in ["EuroSAT"]:
        datamgr         = EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = n_query, **few_shot_params)
        novel_loader     = datamgr.get_data_loader(aug = False )

      elif params.testset in ["CropDisease"]:
        datamgr         = CropDisease_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = n_query, **few_shot_params)
        novel_loader     = datamgr.get_data_loader(aug = False )

      elif params.testset in ["ChestX"]:
        datamgr         = Chest_few_shot.SetDataManager(image_size,  n_eposide = iter_num, n_query = n_query, **few_shot_params)
        novel_loader     = datamgr.get_data_loader(aug = False )

    finetune(novel_loader, n_pseudo=n_pseudo, n_way=params.test_n_way, n_support=params.n_shot)

if __name__=='__main__':
    params = parse_args(script='train')
    #for tmp_testset in ['cub', 'cars', 'places', 'plantae', 'ChestX', 'ISIC', 'EuroSAT', 'CropDisease']:
    #for tmp_testset in ['EuroSAT', 'CropDisease']:
    #for tmp_testset in ['CropDisease']:
    #for tmp_testset in ['EuroSAT', 'plantae']:
    #for tmp_testset in ['ISIC']:
    #for tmp_testset in ['ChestX', 'ISIC']:
    for tmp_testset in ['EuroSAT']:
      params.testset = tmp_testset
      run_single_testset(params)
