import utils
import torch
import numpy as np
import torch.nn as nn
import warnings
from torcheval.metrics import MulticlassAccuracy
from torch.nn.modules.loss import CrossEntropyLoss
from utils import print_progress

warnings.filterwarnings("ignore")


def tester_func(model,dataloader,device,ckpt,num_class,logger):
    model=model.to(device)
    model.eval()

    loss_ce_total   = utils.AverageMeter()

    metric  = MulticlassAccuracy(average="macro", num_classes=num_class).to('cuda')
    loss_ce = CrossEntropyLoss(label_smoothing=0.0)

    total_batchs = len(dataloader['test'])
    loader       = dataloader['test']

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)

            inputs  = inputs.float()
            targets = targets.float()

            outputs = model(inputs)

            if type(outputs)==tuple:
                outputs, targets = outputs[0], outputs[1].argmax(dim=1)

            loss        = loss_ce(outputs, targets.long()) 
            loss_ce_total.update(loss)
   
            predictions = torch.argmax(input=torch.softmax(outputs, dim=1),dim=1).long()
            metric.update(predictions, targets.long())

            print_progress(
                iteration=batch_idx+1,
                total=total_batchs,
                prefix=f'Test Batch {batch_idx+1}/{total_batchs} ',
                suffix=f'loss= {loss_ce_total.avg:.4f} , Accuracy = {100 * metric.compute():.4f}',
                bar_length=45
            )  

        Acc = 100 * metric.compute()
        
        logger.info(f'Final Test ---> Loss = {loss_ce_total.avg:.4f} , Accuracy = {Acc:.2f}') 
