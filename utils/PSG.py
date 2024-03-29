import torch
import torchvision.transforms as transforms
import random

def gamma_correction(x, gamma):
    minv = torch.min(x)
    x = x - minv

    maxv = torch.max(x)
    x = x / maxv

    x = x**gamma
    x = x * maxv + minv
    return x

def random_aug(x):
    #print('x1:', x.size())
    # gamma correction
    if random.random() <= 0.3:
        gamma = random.uniform(1.0, 1.5)
        x = gamma_correction(x, gamma)
    # random erasing with mean value
    mean_v = tuple(x.view(x.size(0), -1).mean(-1))
    re = transforms.RandomErasing(p=0.5, value=mean_v)
    x = re(x)
    # color channel shuffle
    if random.random() <= 0.3:
        l = [0,1,2]
        random.shuffle(l)
        x_c = torch.zeros_like(x)
        x_c[l] = x
        x = x_c
    # horizontal flip or vertical flip
    if random.random() <= 0.5:
        if random.random() <= 0.5:
            x = torch.flip(x, [1])
        else:
            x = torch.flip(x, [2])
    # rotate 90, 180 or 270 degree
    if random.random() <= 0.5:
        degree = [90, 180, 270]
        d = random.choice(degree)
        x = torch.rot90(x, d//90, [1, 2])
    #print('x2:', x.size())
    return x

class PseudoSampleGenerator(object):
    def __init__(self, n_way, n_support, n_pseudo):
        super(PseudoSampleGenerator, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_pseudo = n_pseudo
        self.n_pseudo_per_way = self.n_pseudo//self.n_way

    def generate(self, support_set):  # (5*n_support, 3, 224, 224)
        #default ATA: 1-shot/5-shot
        if(self.n_support<=5):
          times = self.n_pseudo//(self.n_way*self.n_support)+1
          psedo_list = []
          for i in range(support_set.size(0)):
            psedo_list.append(support_set[i])
            for j in range(1, times):
                cur_x = support_set[i]
                cur_x = random_aug(cur_x)
                psedo_list.append(cur_x)
          psedo_set = torch.stack(psedo_list)
          #print('psedo_set:', psedo_set.size())
          psedo_set = psedo_set.reshape([self.n_way, self.n_pseudo_per_way+ self.n_support]+list(psedo_set.size()[1:]))

        # adapt ata to 20/50 shots
        else: 
          #random select 15 support images from 20/50shot
          support_set = support_set.view(self.n_way, self.n_support, 3, 224, 224)
          #print('support_set:', support_set.size())
          perm = torch.randperm(self.n_support)
          idx = perm[:15]
          #print('idx:', idx)
          selected_support_set = support_set[:, idx, :, :, :]
          #print('selected_support_set:', selected_support_set.size())
          selected_support_set = selected_support_set.view(self.n_way*15, 3, 224, 224)             
          # use the selected_support_set to generate pesudo query
          times =1 
          psedo_query_list = []
          for i in range(selected_support_set.size(0)):
            for j in range(0, times):
              cur_x = selected_support_set[i]
              cur_x = random_aug(cur_x)
              psedo_query_list.append(cur_x)
          psedo_query_list = torch.stack(psedo_query_list)
          psedo_query_set = psedo_query_list.view(self.n_way, 15, 3, 224, 224)
          #print('psedo_query_set:', psedo_query_set.size())
          psedo_set = torch.cat((support_set, psedo_query_set), dim = 1)
          #print("psedo_set:", psedo_set.size())
        return psedo_set
