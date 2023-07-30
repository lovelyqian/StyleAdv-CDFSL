import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import random
from options import parse_args
from utils.PSG import PseudoSampleGenerator
from methods.backbone_multiblock import model_dict
from methods.StyleAdv_RN_GNN import StyleAdvGNN
from data.datamgr import SetDataManager
from data import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

#Finetune_LR = 0.001
Finetune_LR = 0.005 
#the finetuning is very sensitive to lr

def finetune(novel_loader, n_pseudo=75, n_way=5, n_support=5):
    iter_num = len(novel_loader)
    acc_all = []

    checkpoint_dir = '%s/checkpoints/%s/best_model.tar' % (params.save_dir, params.resume_dir)
    state = torch.load(checkpoint_dir)['state']
    for ti, (x, _) in enumerate(novel_loader):  # x:(5, 20, 3, 224, 224)
        model = StyleAdvGNN(model_dict[params.model], n_way=n_way, n_support=n_support).cuda()
        model.load_state_dict(state, strict = True)
        x = x.cuda()
        # Finetune components initialization
        xs = x[:, :n_support].reshape(-1, *x.size()[2:])  # (25, 3, 224, 224)
        pseudo_q_genrator = PseudoSampleGenerator(n_way, n_support, n_pseudo)
        loss_fun = nn.CrossEntropyLoss().cuda()
        opt = torch.optim.Adam(model.parameters(), lr = Finetune_LR)
        # Finetune process
        n_query = n_pseudo//n_way
        pseudo_set_y = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
        model.n_query = n_query
        model.train()
        for epoch in range(params.finetune_epoch):
            opt.zero_grad()
            pseudo_set = pseudo_q_genrator.generate(xs)  # (5, n_support+n_query, 3, 224, 224)
            scores = model.set_forward(pseudo_set)  # (5*n_query, 5)
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
            scores = model.set_forward(x)  # (80, 5)
            _, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()  # (80, 1)
            top1_correct = np.sum(topk_ind[:,0]==yq)
            acc = top1_correct*100./(n_way*n_query)
            acc_all.append(acc)
        del scores, topk_labels
        torch.cuda.empty_cache()
        print('Task %d : %4.2f%%, mean Acc: %4.2f'%(ti, acc, np.mean(np.array(acc_all))))

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('Test Acc = %4.2f +- %4.2f%%'%(acc_mean, 1.96*acc_std/np.sqrt(iter_num)))

if __name__=='__main__':
    seed = 0
    print("set seed = %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    params = parse_args('train')

    image_size = 224
    iter_num = 1000
    n_query = 16
    n_pseudo = 75  

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

    import time
    start = time.clock()
    finetune(novel_loader, n_pseudo=n_pseudo, n_way=params.test_n_way, n_support=params.n_shot)
    end = time.clock()
    print('Running time: %s Seconds: %s Min: %s Min per epoch'%(end-start, (end-start)/60, (end-start)/60/iter_num))
    
