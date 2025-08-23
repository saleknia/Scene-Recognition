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

warnings.filterwarnings("ignore")

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

freq = np.array([ 672.,  747.,  607.,  699.,  241.,  925.,  750.,  638.,  871.,
        596.,  238.,  294.,  207.,  435.,  588.,  169.,  491.,   82.,
        711.,  752.,  314.,  630.,  555.,  451.,  555.,  108.,  441.,
        331.,   85.,  448.,  118., 1032.,  388.,  186.,  263.,  277.,
        394.,  567.,  342.,  109., 1771., 1700., 1888.,  908., 1835.,
       1586.,  214.,  635.,  632.,  464.,  324.,  425.,  185.,  393.,
       1253.,  338., 1712.,  297.,  558., 1361.,  351.,  594.,  861.,
        135., 1145.,  171.,  413.,  277.,  519.,  158.,  285., 1566.,
         81.,   50., 6709., 2466., 2109.,  631., 1056.,  602.,  154.,
        679., 1239.,  417.,  280.,  876.,  388., 1579., 8412., 5541.,
        504., 4947., 1536., 7609.,  823., 1198.,  822.,  366.,  510.,
        144.,  853.,  370.])

total = 14340  
neg_counts = total - freq
pos_weight = torch.tensor(neg_counts / (freq + 1e-6), dtype=torch.float32).cuda()

# ----------------------------
# Trainer Function
# ----------------------------
def trainer_func(epoch_num, model, dataloader, optimizer, device, ckpt, num_class, lr_scheduler, logger):
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    
    model = model.to(device)
    model.train()

    loss_total     = utils.AverageMeter() 
    loss_att_total = utils.AverageMeter() 
    soft_acc_total = utils.AverageMeter()

    # BCE loss with per-attribute pos_weight
    loss_att_func = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    total_batchs = len(dataloader['train'])
    loader       = dataloader['train'] 

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, labels = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)

        ###################################################################
        # SUN mask: ignore attributes with single votes
        binary_labels = (labels >= 0.6).float()
        mask = ((labels >= 0.6) | (labels <= 0.1)).float()
        num_valid_labels = mask.sum()

        # Element-wise BCE with pos_weight
        loss_per_element = loss_att_func(outputs, binary_labels)
        masked_loss      = loss_per_element * mask
        loss_att         = masked_loss.sum() / torch.clamp(num_valid_labels, min=1.0)

        loss_att_total.update(loss_att.item(), n=num_valid_labels.item())   
        loss_total.update(loss_att.item())
        ###################################################################   

        optimizer.zero_grad()
        loss_att.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step() 

        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            correct = ((preds == binary_labels).float() * mask).sum().item()
            batch_accuracy = correct / num_valid_labels.item() if num_valid_labels > 0 else 0
            soft_acc_total.update(batch_accuracy, n=num_valid_labels.item())

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'Loss = {loss_total.avg:.4f}, att_Loss = {loss_att_total.avg:.4f}, SoftAcc = {100 * soft_acc_total.avg:.2f}',   
            bar_length=45
        )  

    logger.info(
        f'Epoch: {epoch_num} ---> Train , Loss = {loss_total.avg:.4f}, '
        f'SoftAcc = {100 * soft_acc_total.avg:.2f}, '
        f'lr = {optimizer.param_groups[0]["lr"]}'
    )

    # ----------------- Validation -----------------
    if epoch_num % 1 == 0:
        model.eval()
        all_probs = []
        all_labels = []

        val_loader = dataloader['valid']

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # SUN mask for validation
        valid_mask = (all_labels >= 0.6) | (all_labels <= 0.1)
        binary_labels_all = (all_labels >= 0.6).float()

        aps = torch.zeros(all_labels.shape[1])
        for attr_idx in range(all_labels.shape[1]):
            mask_i = valid_mask[:, attr_idx]
            if not mask_i.any():
                continue
            scores_i = all_probs[mask_i, attr_idx]
            labels_i = binary_labels_all[mask_i, attr_idx]
            if labels_i.sum() == 0:
                continue
            ap = average_precision_score(labels_i.numpy().astype(np.int32),
                                         scores_i.numpy())
            aps[attr_idx] = ap

        valid_aps = aps[aps > 0]
        mean_ap = valid_aps.mean().item() if len(valid_aps) > 0 else 0
        print(mean_ap)
        logger.info(f'** Epoch: {epoch_num} ---> Validation mAP (thr=0.5): {100 * mean_ap:.2f} **')

        # Save checkpoint based on validation mAP
        if ckpt is not None:
            ckpt.save_best(acc=100 * mean_ap, epoch=epoch_num, net=model)

        model.train()

    # # --- START EVALUATION FUNCTION ---
    # model.eval()
    # all_probs = []
    # all_labels = []

    # val_loader = dataloader['valid'] # Assuming your dataloader dict has a 'val' key

    # metric_val = MulticlassAccuracy(average="macro", num_classes=num_cat).to(device)

    # with torch.no_grad():
    #     for inputs, (labels, categories) in val_loader:

    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         categories = categories.to(device)

    #         outputs_att, outputs_cat = model(inputs)
    #         ############################################################################
    #         probs = torch.sigmoid(outputs_att)
    #         all_probs.append(probs.cpu())
    #         all_labels.append(labels.cpu()) # Keep original vote fractions for filtering
    #         ############################################################################
    #         predictions = torch.argmax(input=torch.softmax(outputs_cat, dim=1),dim=1).long()
    #         metric_val.update(predictions, categories.long())     

    # all_probs = torch.cat(all_probs, dim=0).numpy()
    # all_labels = torch.cat(all_labels, dim=0).numpy()

    # aps = [] # List to store AP for each attribute
    # for attr_idx in range(all_labels.shape[1]): # Loop over each attribute
    #     attr_scores = all_probs[:, attr_idx]
    #     attr_votes = all_labels[:, attr_idx]

    #     # Create mask for valid examples: definitively present or absent
    #     valid_mask = (attr_votes >= 0.6) | (attr_votes <= 0.1)
    #     valid_scores = attr_scores[valid_mask]
    #     valid_labels = (attr_votes[valid_mask] >= 0.6).astype(np.int32)

    #     # Only calculate AP if there are positive examples in the valid set
    #     if np.sum(valid_labels) > 0:
    #         ap = average_precision_score(valid_labels, valid_scores)
    #         aps.append(ap)
    #     else:
    #         # If no positives, skip to avoid division by zero
    #         print(f"Warning: Attribute {attr_idx} has no positive examples. Skipping.")

    # # Calculate the mean Average Precision (mAP)
    # mean_ap = np.mean(aps) if aps else 0
    # # --- END EVALUATION FUNCTION ---



