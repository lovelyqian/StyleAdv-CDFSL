import math
import sys
import warnings
from typing import Iterable, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

#import pmf_utils.deit_util as utils
#from pmf_utils import AverageMeter, to_device
from utils import AverageMeter, to_device
import utils.deit_util as utils

import numpy as np

#from methods.meta_template_StyleAdvIncrem_v10_epsilonFromList_RandomStartFGSM_20220501 import consistency_loss
#from methods.meta_template_StyleAdv_RN_GNN import consistency_loss
from methods.tool_func import consistency_loss

def train_one_epoch_styleAdv(data_loader: Iterable,
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    device: torch.device,
                    loss_scaler = None,
                    fp16: bool = False,
                    max_norm: float = 0, # clip_grad
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    writer: Optional[SummaryWriter] = None,
                    set_training_mode=True):

    global_step = epoch * len(data_loader)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    model.train(set_training_mode)

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch = to_device(batch, device)
        SupportTensor, SupportLabel, QueryTensor, QueryLabel, GlobalID_S, GlobalID_Q = batch
        #print('SupportTensor:', SupportTensor.size(), 'SupportLabel:', SupportLabel, 'x:', x.size(), 'y:', y.size())

        epsilon_list = [0.8, 0.08, 0.008] 
        # forward
        with torch.cuda.amp.autocast(fp16):
            #output = model(SupportTensor, SupportLabel, x)
            scores_fsl_ori, loss_fsl_ori, scores_cls_ori, loss_cls_ori, scores_fsl_adv, loss_fsl_adv, scores_cls_adv, loss_cls_adv = model.set_forward_loss_StyAdv(SupportTensor,QueryTensor,SupportLabel, QueryLabel, GlobalID_S,GlobalID_Q, epsilon_list) 
        if(scores_fsl_ori.equal(scores_fsl_adv)):
          loss_fsl_KL = 0
        else:
          loss_fsl_KL = consistency_loss(scores_fsl_ori, scores_fsl_adv, 'KL3')
        if(scores_cls_ori.equal(scores_cls_adv)):
          loss_cls_KL = 0
        else:
          loss_cls_KL = consistency_loss(scores_cls_ori, scores_cls_adv,'KL3')

        k1, k2, k3, k4, k5, k6 = 1, 1, 1, 1, 0, 0
        loss = k1 * loss_fsl_ori + k2 * loss_fsl_adv + k3 * loss_fsl_KL + k4 * loss_cls_ori + k5 * loss_cls_adv + k6 * loss_cls_KL
        #print('loss_fsl_ori:', loss_fsl_ori, 'loss_fsl_adv:', loss_fsl_adv, 'loss_fsl_KL:', loss_fsl_KL, 'loss_cls_ori:', loss_cls_ori, 'loss_cls_adv:',loss_cls_adv, 'loss_cls_adv')
        #output = output.view(QueryTensor.shape[0] * QueryTensor.shape[1], -1)
        #QueryLabel = QueryLabel.view(-1)
        #loss = criterion(output, QueryLabel)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if fp16:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
        metric_logger.update(n_ways=SupportLabel.max()+1)
        metric_logger.update(n_imgs=SupportTensor.shape[1] + QueryTensor.shape[1])

        # tensorboard
        if utils.is_main_process() and global_step % print_freq == 0:
            writer.add_scalar("train/loss", scalar_value=loss_value, global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=lr, global_step=global_step)

        global_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(data_loaders, model, criterion, device, seed=None, ep=None):
    if isinstance(data_loaders, dict):
        test_stats_lst = {}
        test_stats_glb = {}

        for j, (source, data_loader) in enumerate(data_loaders.items()):
            print(f'* Evaluating {source}:')
            seed_j = seed + j if seed else None
            test_stats = _evaluate(data_loader, model, criterion, device, seed_j)
            test_stats_lst[source] = test_stats
            test_stats_glb[source] = test_stats['acc1']

        # apart from individual's acc1, accumulate metrics over all domains to compute mean
        for k in test_stats_lst[source].keys():
            test_stats_glb[k] = torch.tensor([test_stats[k] for test_stats in test_stats_lst.values()]).mean().item()

        return test_stats_glb
    elif isinstance(data_loaders, torch.utils.data.DataLoader): # when args.eval = True
        return _evaluate(data_loaders, model, criterion, device, seed, ep)
    else:
        warnings.warn(f'The structure of {data_loaders} is not recognizable.')
        return _evaluate(data_loaders, model, criterion, device, seed)


@torch.no_grad()
def _evaluate(data_loader, model, criterion, device, seed=None, ep=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('acc1', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    metric_logger.add_meter('acc5', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    # added for debug
    #metric_logger.add_meter('loss', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if seed is not None:
        data_loader.generator.manual_seed(seed)

    for ii, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if ep is not None:
            if ii > ep:
                break

        batch = to_device(batch, device)
        SupportTensor, SupportLabel, x, y = batch
        #print('SupportTensor:', SupportTensor.size(), 'SupportLabel:', SupportLabel, 'x:', x.size(), 'y:', y.size())

        # compute output
        with torch.cuda.amp.autocast():
            output = model(SupportTensor, SupportLabel, x)

        output = output.view(x.shape[0] * x.shape[1], -1)
        y = y.view(-1)
        loss = criterion(output, y)
        acc1, acc5 = accuracy(output, y, topk=(1, 5))

        batch_size = x.shape[0]
        metric_logger.update(loss=loss.item())
        # for debug
        #metric_logger.meters['loss'].update(loss.item(), n=batch_size)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.update(n_ways=SupportLabel.max()+1)
        metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # initial
    #print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #      .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
 
    ret_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    ret_dict['acc_std'] = metric_logger.meters['acc1'].std
    print('ret dict:', ret_dict['acc_std'], metric_logger.meters['acc1'], metric_logger.meters['acc1'].std)

    ''' 
    # debug for test BSCDFSL
    ret_dict['acc_std'] = metric_logger.meters['acc1'].std
    '''
    return ret_dict
