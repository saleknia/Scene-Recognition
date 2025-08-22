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

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

# ----------------------------
# Pos/Neg Split BCE Loss
# ----------------------------
def bce_pos_neg_separate(logits, labels, mask=None):
    """
    logits: [B, C] raw outputs
    labels: [B, C] binary 0/1
    mask: [B, C] optional mask to ignore uncertain labels
    """
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')

    if mask is not None:
        bce = bce * mask

    # Separate positives and negatives
    pos_mask = labels == 1
    neg_mask = labels == 0

    if mask is not None:
        pos_mask = pos_mask * (mask.bool())
        neg_mask = neg_mask * (mask.bool())

    # Per-sample normalization
    loss_pos = (bce * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1.0)
    loss_neg = (bce * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1.0)

    loss = loss_pos + loss_neg
    return loss.mean()


# ----------------------------
# Trainer Function (fixed threshold 0.5)
# ----------------------------
def trainer_func(epoch_num, model, dataloader, optimizer, device, ckpt, num_class, lr_scheduler, logger):
    print(f"Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]['lr']}")

    model = model.to(device)
    model.train()

    loss_total     = utils.AverageMeter()
    loss_att_total = utils.AverageMeter()
    soft_acc_total = utils.AverageMeter()

    total_batchs = len(dataloader["train"])
    loader = dataloader["train"]

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, labels = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # SUN masking logic
        binary_labels = (labels >= 0.6).float()
        mask = ((labels >= 0.6) | (labels <= 0.1)).float()

        # Pos/Neg split BCE
        loss_att = bce_pos_neg_separate(outputs, binary_labels, mask)

        loss_total.update(loss_att.item())
        loss_att_total.update(loss_att.item())

        optimizer.zero_grad()
        loss_att.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Soft accuracy for logging
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()  # fixed threshold
            correct = ((preds == binary_labels).float() * mask).sum().item()
            num_valid_labels = mask.sum().item()
            batch_accuracy = correct / num_valid_labels if num_valid_labels > 0 else 0
            soft_acc_total.update(batch_accuracy, n=num_valid_labels)

        # Progress bar
        print_progress(
            iteration=batch_idx + 1,
            total=total_batchs,
            prefix=f"Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ",
            suffix=f"Loss = {loss_total.avg:.4f}, att_Loss = {loss_att_total.avg:.4f}, SoftAcc = {100 * soft_acc_total.avg:.2f}",
            bar_length=45,
        )

    logger.info(
        f"Epoch: {epoch_num} ---> Train , Loss = {loss_total.avg:.4f}, "
        f"SoftAcc = {100 * soft_acc_total.avg:.2f}, "
        f"lr = {optimizer.param_groups[0]['lr']}"
    )

    # ----------------------------
    # Validation
    # ----------------------------
    model.eval()
    all_probs = []
    all_labels = []

    val_loader = dataloader["valid"]
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    valid_mask = (all_labels >= 0.6) | (all_labels <= 0.1)
    binary_labels_all = (all_labels >= 0.6).float()

    # --- Compute AP per attribute
    aps = torch.zeros(all_labels.shape[1])
    for attr_idx in range(all_labels.shape[1]):
        mask_i = valid_mask[:, attr_idx]
        if not mask_i.any():
            continue
        scores_i = all_probs[mask_i, attr_idx]
        labels_i = binary_labels_all[mask_i, attr_idx]
        if labels_i.sum() == 0:
            continue
        ap = average_precision_score(labels_i.numpy().astype(int),
                                     scores_i.numpy())
        aps[attr_idx] = ap

    valid_aps = aps[aps > 0]
    mean_ap = valid_aps.mean().item() if len(valid_aps) > 0 else 0

    # --- Compute accuracy with fixed 0.5 threshold
    preds = (all_probs >= 0.5).float()
    correct = ((preds == binary_labels_all).float() * valid_mask).sum().item()
    total_valid = valid_mask.sum().item()
    acc = correct / total_valid if total_valid > 0 else 0

    logger.info(f"** Epoch {epoch_num} ---> Validation mAP: {100*mean_ap:.2f}, Accuracy: {100*acc:.2f} **")

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



