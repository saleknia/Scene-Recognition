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

from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import numpy as np

def batch_average_precision_torch(scores, labels):
    """
    Calculate Average Precision for a batch of predictions in parallel.
    scores: Tensor of shape [num_samples, num_attributes]
    labels: Tensor of shape [num_samples, num_attributes] (binary)
    returns: Tensor of shape [num_attributes] with AP scores
    """
    num_attributes = scores.shape[1]
    aps = torch.zeros(num_attributes, device=scores.device)
    
    for attr_idx in range(num_attributes):
        attr_scores = scores[:, attr_idx]
        attr_labels = labels[:, attr_idx]
        
        if attr_labels.sum() == 0:
            continue
            
        # Sort by scores descending
        descending_indices = torch.argsort(attr_scores, descending=True)
        sorted_scores = attr_scores[descending_indices]
        sorted_labels = attr_labels[descending_indices]
        
        # Calculate precision at each threshold
        tp_cumsum = torch.cumsum(sorted_labels, dim=0)
        precision_at_k = tp_cumsum / (torch.arange(1, len(sorted_labels) + 1, device=scores.device, dtype=torch.float32))
        
        # Calculate AP
        ap = torch.sum(precision_at_k * sorted_labels) / torch.clamp(torch.sum(sorted_labels), min=1e-6)
        aps[attr_idx] = ap
    
    return aps

def trainer_func(epoch_num, model, dataloader, optimizer, device, ckpt, num_class, lr_scheduler, logger):
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    
    model = model.to(device)
    model.train()

    num_att, num_cat = num_class

    loss_total     = utils.AverageMeter() 
    loss_cat_total = utils.AverageMeter()
    loss_att_total = utils.AverageMeter() 
    soft_acc_total = utils.AverageMeter()

    metric_train = MulticlassAccuracy(average="macro", num_classes=num_cat).to(device)
    
    loss_cat_func = CrossEntropyLoss(label_smoothing=0.0)
    loss_att_func = nn.BCEWithLogitsLoss(reduction='none')

    total_batchs = len(dataloader['train'])
    loader       = dataloader['train'] 

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, (labels, categories) = inputs, targets

        inputs = inputs.to(device)
        labels = labels.to(device)
        categories = categories.to(device)
        
        outputs_att, outputs_cat = model(inputs)

        ###################################################################
        binary_labels = (labels >= 0.6).float()
        mask = ((labels >= 0.6) | (labels <= 0.1)).float()
        num_valid_labels = mask.sum()

        loss_per_element = loss_att_func(outputs_att, binary_labels)
        masked_loss      = loss_per_element * mask
        loss_att         = masked_loss.sum() / torch.clamp(num_valid_labels, min=1.0)

        loss_att_total.update(loss_att.item(), n=num_valid_labels.item())
        ###################################################################
        predictions = torch.argmax(input=torch.softmax(outputs_cat, dim=1),dim=1).long()
        loss_cat    = 0.05 * loss_cat_func(outputs_cat, categories.long())      
        loss_cat_total.update(loss_cat.item())
        metric_train.update(predictions, categories.long())     
        ###################################################################
        loss = loss_cat + loss_att
        loss_total.update(loss.item())
        ###################################################################   

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step() 

        with torch.no_grad():
            probs = torch.sigmoid(outputs_att)
            preds = (probs >= 0.5).float()
            correct = ((preds == binary_labels).float() * mask).sum().item()
            batch_accuracy = correct / num_valid_labels.item() if num_valid_labels > 0 else 0
            soft_acc_total.update(batch_accuracy, n=num_valid_labels.item())

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'Loss = {loss_total.avg:.4f}, cat_Loss = {loss_cat_total.avg:.4f}, att_Loss = {loss_att_total.avg:.4f}, SoftAcc = {100 * soft_acc_total.avg:.2f}, CatAcc = {100 * metric_train.compute():.2f}',   
            bar_length=45
        )  

    logger.info(
        f'Epoch: {epoch_num} ---> Train , Loss = {loss_total.avg:.4f}, '
        f'SoftAcc = {100 * soft_acc_total.avg:.2f}, '
        f'CatAcc  = {100 * metric_train.compute():.2f}, '
        f'lr = {optimizer.param_groups[0]["lr"]}'
    )
    # --- ULTRA-FAST VALIDATION ---
    model.eval()
    metric_val = MulticlassAccuracy(average="macro", num_classes=num_cat).to(device)
    
    # Pre-allocate tensors to avoid appending
    num_val_samples = len(dataloader['valid'].dataset)
    all_probs = torch.zeros(num_val_samples, num_att, device=device)
    all_labels = torch.zeros(num_val_samples, num_att, device=device)
    all_cat_preds = torch.zeros(num_val_samples, dtype=torch.long, device=device)
    all_cat_labels = torch.zeros(num_val_samples, dtype=torch.long, device=device)
    
    current_idx = 0
    with torch.no_grad():
        for inputs, (labels, categories) in dataloader['valid']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            categories = categories.to(device)
            
            batch_size = inputs.size(0)
            outputs_att, outputs_cat = model(inputs)
            
            # Store predictions and labels
            end_idx = current_idx + batch_size
            all_probs[current_idx:end_idx] = torch.sigmoid(outputs_att)
            all_labels[current_idx:end_idx] = labels
            
            # Category predictions
            cat_preds = torch.argmax(torch.softmax(outputs_cat, dim=1), dim=1)
            all_cat_preds[current_idx:end_idx] = cat_preds
            all_cat_labels[current_idx:end_idx] = categories.long()
            
            metric_val.update(cat_preds, categories.long())
            current_idx = end_idx

    # Calculate mAP in parallel
    valid_mask = (all_labels >= 0.6) | (all_labels <= 0.1)
    binary_labels = (all_labels >= 0.6).float()
    
    # Apply mask
    masked_probs = all_probs * valid_mask.float()
    masked_binary_labels = binary_labels * valid_mask.float()
    
    # Calculate AP for each attribute
    aps = batch_average_precision_torch(masked_probs, masked_binary_labels)
    
    # Filter out attributes with no valid examples
    valid_aps = aps[aps > 0]
    mean_ap = valid_aps.mean().item() if len(valid_aps) > 0 else 0
    val_acc = metric_val.compute().item()
    logger.info(f'** Epoch: {epoch_num} ---> Validation mAP: {100 * mean_ap:.2f}, Validation Category Accuracy: {100 * metric_val.compute():.2f} **')

    # Save checkpoint based on the validation mAP, not training loss
    if ckpt is not None:
        ckpt.save_best(acc=100 * mean_ap, epoch=epoch_num, net=model)

    # Set model back to training mode for the next epoch
    model.train()

    # # --- START OPTIMIZED EVALUATION FUNCTION ---
    # model.eval()
    # all_probs = []
    # all_labels = []

    # val_loader = dataloader['valid']
    # metric_val = MulticlassAccuracy(average="macro", num_classes=num_cat).to(device)

    # with torch.no_grad():
    #     for inputs, (labels, categories) in val_loader:
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         categories = categories.to(device)

    #         outputs_att, outputs_cat = model(inputs)
    #         probs = torch.sigmoid(outputs_att)
    #         all_probs.append(probs.cpu())
    #         all_labels.append(labels.cpu())

    #         predictions = torch.argmax(input=torch.softmax(outputs_cat, dim=1), dim=1).long()
    #         metric_val.update(predictions, categories.long())

    # # Convert to PyTorch tensors first for faster GPU-enabled operations if available, then to numpy.
    # all_probs = torch.cat(all_probs, dim=0)  # Keep as tensor for a moment
    # all_labels = torch.cat(all_labels, dim=0)

    # # --- VECTORIZED MASK CREATION ---
    # # Create the valid mask for ALL attributes simultaneously
    # # Shape: [num_samples, num_attributes]
    # valid_mask = (all_labels >= 0.6) | (all_labels <= 0.1)

    # # Create binary labels for ALL attributes
    # binary_labels_all = (all_labels >= 0.6).float()

    # # Initialize a tensor to hold APs for all attributes
    # aps = torch.zeros(all_labels.shape[1])  # [num_attributes]

    # # --- VECTORIZED AP CALCULATION ---
    # # We still need a loop, but we can precompute everything for each attribute very efficiently.
    # for attr_idx in range(all_labels.shape[1]):
    #     # Use the precomputed mask for this attribute
    #     mask_i = valid_mask[:, attr_idx]
    #     # If there are no valid examples, skip and AP remains 0.
    #     if not mask_i.any():
    #         # print(f"Warning: Attribute {attr_idx} has no valid examples. Skipping.")
    #         continue

    #     # Apply the mask using boolean indexing
    #     scores_i = all_probs[mask_i, attr_idx]
    #     labels_i = binary_labels_all[mask_i, attr_idx]

    #     # Check if there are positive examples to avoid sklearn error
    #     if labels_i.sum() == 0:
    #         # print(f"Warning: Attribute {attr_idx} has no positive examples. Skipping.")
    #         continue

    #     # Move to CPU numpy for sklearn (this is the unavoidable step, but it's now on a pre-filtered subset)
    #     scores_i_np = scores_i.numpy()
    #     labels_i_np = labels_i.numpy().astype(np.int32)

    #     # Calculate AP for this attribute
    #     ap = average_precision_score(labels_i_np, scores_i_np)
    #     aps[attr_idx] = ap

    # # Calculate the mean Average Precision (mAP), ignoring attributes that were skipped (with 0 AP)
    # valid_aps = aps[aps > 0]
    # mean_ap = valid_aps.mean().item() if len(valid_aps) > 0 else 0
    # # --- END OPTIMIZED EVALUATION FUNCTION ---
    
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



