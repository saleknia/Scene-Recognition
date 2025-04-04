import utils
import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import warnings
import numpy as np
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy
from utils import print_progress

warnings.filterwarnings("ignore")

def valid_func(epoch_num,model,dataloader,device,ckpt,num_class,logger):
    
    model = model.to('cuda')
    model = model.eval()

    loss_ce_total   = utils.AverageMeter()
    
    metric  = MulticlassAccuracy(average="macro", num_classes=num_class).to('cuda')
    loss_ce = CrossEntropyLoss(label_smoothing=0.0)

    total_batchs = len(dataloader['valid'])
    loader = dataloader['valid']

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)

            inputs  = inputs.float()
            targets = targets.float()

            outputs = model(inputs)
            
            loss    = loss_ce(outputs, targets.long()) 

            loss_ce_total.update(loss)
   
            predictions = torch.argmax(input=torch.softmax(outputs, dim=1),dim=1).long()
            metric.update(predictions, targets.long())

            print_progress(
                iteration=batch_idx+1,
                total=total_batchs,
                prefix=f'Valid {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
                suffix=f'loss = {loss_ce_total.avg:.4f} , loss_ce = {loss_ce_total.avg:.4f}, Accuracy = {100 * metric.compute():.4f}',         
                bar_length=45
            )  

    Acc = 100 * metric.compute()

    logger.info(f'Epoch: {epoch_num} ---> Train , Loss = {loss_ce_total.avg:.4f}, Accuracy = {Acc:.2f}')

    # Save checkpoint
    if ckpt is not None:
        ckpt.save_best(acc=Acc, epoch=epoch_num, net=model)
