import utils
import torch
import numpy as np
import torch.nn as nn
import warnings
from torcheval.metrics import MulticlassAccuracy
from torch.nn.modules.loss import CrossEntropyLoss
from utils import print_progress

warnings.filterwarnings("ignore")

def tester_func(model, dataloader, device, ckpt, num_class, logger):
    model = model.to(device)
    model.eval()

    loss_ce_total = utils.AverageMeter()

    # Change here: Use average=None to get per-class accuracies
    metric_macro = MulticlassAccuracy(average="macro", num_classes=num_class).to('cuda')
    metric_per_class = MulticlassAccuracy(average=None, num_classes=num_class).to('cuda')  # Add this
    
    loss_ce = CrossEntropyLoss(label_smoothing=0.0)

    total_batchs = len(dataloader['test'])
    loader = dataloader['test']

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            targets = targets.float()

            outputs = model(inputs)

            if type(outputs) == tuple:
                outputs, targets = outputs[0], outputs[1].argmax(dim=1)

            loss = loss_ce(outputs, targets.long())
            loss_ce_total.update(loss)

            predictions = torch.argmax(input=torch.softmax(outputs, dim=1), dim=1).long()
            
            # Update both metrics
            metric_macro.update(predictions, targets.long())
            metric_per_class.update(predictions, targets.long())

            print_progress(
                iteration=batch_idx+1,
                total=total_batchs,
                prefix=f'Test Batch {batch_idx+1}/{total_batchs} ',
                suffix=f'loss= {loss_ce_total.avg:.4f} , Accuracy = {100 * metric_macro.compute():.4f}',
                bar_length=45
            )

        Acc_macro = 100 * metric_macro.compute()
        Acc_per_class = 100 * metric_per_class.compute()  # This is a tensor of shape [num_class]
        
        # Log per-class accuracies
        logger.info(f'Final Test ---> Loss = {loss_ce_total.avg:.4f} , Macro Accuracy = {Acc_macro:.2f}')
        logger.info('Per-class accuracies:')
        
        for class_idx in range(num_class):
            logger.info(f'  Class {class_idx}: {Acc_per_class[class_idx]:.2f}%')
        
        # Also print to console
        print('\n' + '='*50)
        print(f'Per-class Accuracies:')
        for class_idx in range(num_class):
            print(f'  Class {class_idx}: {Acc_per_class[class_idx]:.2f}%')
        print('='*50)

        return Acc_macro, Acc_per_class
# import utils
# import torch
# import numpy as np
# import torch.nn as nn
# import warnings
# from torcheval.metrics import MulticlassAccuracy
# from torch.nn.modules.loss import CrossEntropyLoss
# from utils import print_progress

# warnings.filterwarnings("ignore")


# def tester_func(model,dataloader,device,ckpt,num_class,logger):
#     model=model.to(device)
#     model.eval()

#     loss_ce_total   = utils.AverageMeter()

#     metric  = MulticlassAccuracy(average="macro", num_classes=num_class).to('cuda')
#     loss_ce = CrossEntropyLoss(label_smoothing=0.0)

#     total_batchs = len(dataloader['test'])
#     loader       = dataloader['test']

#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(loader):

#             inputs, targets = inputs.to(device), targets.to(device)

#             inputs  = inputs.float()
#             targets = targets.float()

#             outputs = model(inputs)

#             if type(outputs)==tuple:
#                 outputs, targets = outputs[0], outputs[1].argmax(dim=1)

#             loss        = loss_ce(outputs, targets.long()) 
#             loss_ce_total.update(loss)
   
#             predictions = torch.argmax(input=torch.softmax(outputs, dim=1),dim=1).long()
#             metric.update(predictions, targets.long())

#             print_progress(
#                 iteration=batch_idx+1,
#                 total=total_batchs,
#                 prefix=f'Test Batch {batch_idx+1}/{total_batchs} ',
#                 suffix=f'loss= {loss_ce_total.avg:.4f} , Accuracy = {100 * metric.compute():.4f}',
#                 bar_length=45
#             )  

#         Acc = 100 * metric.compute()
        
#         logger.info(f'Final Test ---> Loss = {loss_ce_total.avg:.4f} , Accuracy = {Acc:.2f}') 
