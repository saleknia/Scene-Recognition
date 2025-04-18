import utils
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import print_progress
import torch.nn.functional as F
import warnings
from torch.autograd import Variable
from valid import valid_func
from torcheval.metrics import MulticlassAccuracy
from torch.nn.modules.loss import CrossEntropyLoss
# labels = torch.load('/content/Scene-Recognition/labels.pt').cuda()
mapping = [0, 1, 0, 1, 0, 2, 2, 1, 0, 1, 0, 2, 0, 2, 0, 1, 1, 2, 0, 0, 1, 2,
       1, 0, 0, 1, 2, 1, 1, 1, 2, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 1, 0, 2,
       0, 2, 0, 1, 2, 2, 2, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 1, 2, 1,
       1]
warnings.filterwarnings("ignore")

def trainer_func(epoch_num,model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,logger):
    # torch.autograd.set_detect_anomaly(True)
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    
    model = model.to('cuda')
    model.train()

    loss_ce_total   = utils.AverageMeter()

    # accuracy = utils.AverageMeter()
    metric = MulticlassAccuracy(average="macro", num_classes=num_class).to('cuda')
    # accuracy = mAPMeter()
    loss_ce = CrossEntropyLoss(label_smoothing=0.0)

    total_batchs = len(dataloader['train'])
    loader       = dataloader['train'] 

    base_iter    = (epoch_num-1) * total_batchs
    iter_num     = base_iter

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)

        targets = targets.float()
        outputs = model(inputs)

        if type(outputs)==tuple:
            outputs, aux = outputs[0], outputs[1]
            goals = torch.tensor([mapping[x] for x in targets.long()]).long().cuda()
            loss  = loss_ce(outputs, targets.long()) + (loss_ce(aux, goals.long()) * 0.5)
        else:
            loss = loss_ce(outputs, targets.long())

        loss_ce_total.update(loss)
        predictions = torch.argmax(input=torch.softmax(outputs, dim=1),dim=1).long()
        metric.update(predictions, targets.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step() 

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'CE_Loss = {loss_ce_total.avg:.4f} , Accuracy = {100 * metric.compute():.4f}',   
            # suffix=f'CE_loss = {loss_ce_total.avg:.4f} , disparity_loss = {loss_disparity_total.avg:.4f} , Accuracy = {100 * accuracy.value().item():.4f}',                 
            # suffix=f'CE_loss = {loss_ce_total.avg:.4f} , disparity_loss = {loss_disparity_total.avg:.4f} , Accuracy = {100 * accuracy.avg:.4f}',   
            bar_length=45
        )  
  
    Acc = 100 * metric.compute()
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss = {loss_ce_total.avg:.4f}, Accuracy = {Acc:.2f} , lr = {optimizer.param_groups[0]["lr"]}')
    # valid_func(epoch_num,copy.deepcopy(model),dataloader,device,ckpt,num_class,logger)
    if ckpt is not None:
       ckpt.save_best(acc=Acc, epoch=epoch_num, net=model)

