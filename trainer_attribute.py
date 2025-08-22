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
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

# ----------------------------
# Loss functions
# ----------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, mask=None):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if mask is not None:
            focal = focal * mask

        if self.reduction == "mean":
            denom = mask.sum() if mask is not None else torch.numel(focal)
            return focal.sum() / denom.clamp(min=1.0)
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal


class AsymmetricLoss(nn.Module):
    """ Ridnik et al. 'Asymmetric Loss For Multi-Label Classification' """
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        xs_pos = probs
        xs_neg = 1 - probs

        # Clip easy negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic loss
        loss = targets * torch.log(xs_pos.clamp(min=self.eps)) + \
               (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        # Focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = xs_pos * targets + xs_neg * (1 - targets)
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss *= (1 - pt) ** one_sided_gamma

        loss = -loss

        if mask is not None:
            loss = loss * mask

        return loss.sum() / mask.sum().clamp(min=1.0) if mask is not None else loss.mean()


# ----------------------------
# Threshold finder
# ----------------------------

def find_optimal_thresholds(probs, labels, valid_mask, step=0.05):
    """
    Compute per-attribute thresholds maximizing F1.
    """
    num_attrs = labels.shape[1]
    thresholds = torch.full((num_attrs,), 0.5)  # default

    for attr_idx in range(num_attrs):
        mask_i = valid_mask[:, attr_idx]
        if not mask_i.any():
            continue

        scores_i = probs[mask_i, attr_idx].numpy()
        labels_i = labels[mask_i, attr_idx].numpy().astype(int)

        if labels_i.sum() == 0 or labels_i.sum() == len(labels_i):
            continue

        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.0, 1.01, step):
            preds = (scores_i >= thr).astype(int)
            f1 = f1_score(labels_i, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        thresholds[attr_idx] = best_thr

    return thresholds


# ----------------------------
# Trainer Function
# ----------------------------

def trainer_func(epoch_num, model, dataloader, optimizer, device, ckpt, num_class, lr_scheduler, logger,
                 loss_type="asl"):
    """
    loss_type: "bce", "focal", or "asl"
    """

    print(f"Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]['lr']}")

    model = model.to(device)
    model.train()

    loss_total     = utils.AverageMeter()
    loss_att_total = utils.AverageMeter()
    soft_acc_total = utils.AverageMeter()

    # Pick loss
    if loss_type == "bce":
        loss_func = nn.BCEWithLogitsLoss(reduction="none")
    elif loss_type == "focal":
        loss_func = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    elif loss_type == "asl":
        loss_func = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    total_batchs = len(dataloader["train"])
    loader = dataloader["train"]

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, labels = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # SUN masking logic
        binary_labels = (labels >= 0.6).float()
        mask = ((labels >= 0.6) | (labels <= 0.1)).float()

        if loss_type == "bce":
            loss_per_element = loss_func(outputs, binary_labels)
            masked_loss = loss_per_element * mask
            loss_att = masked_loss.sum() / torch.clamp(mask.sum(), min=1.0)
        else:
            loss_att = loss_func(outputs, binary_labels, mask)

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
            preds = (probs >= 0.5).float()
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

    # Epoch summary
    logger.info(
        f"Epoch: {epoch_num} ---> Train , Loss = {loss_total.avg:.4f}, "
        f"SoftAcc = {100 * soft_acc_total.avg:.2f}, "
        f"lr = {optimizer.param_groups[0]['lr']}"
    )

    # ----------------------------
    # Validation
    # ----------------------------
    if epoch_num % 1 == 0:
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

        # --- Compute per-attribute thresholds ---
        thresholds = find_optimal_thresholds(all_probs, binary_labels_all, valid_mask, step=0.05)
        preds_opt = torch.zeros_like(all_probs)
        for attr_idx in range(all_labels.shape[1]):
            thr = thresholds[attr_idx].item()
            preds_opt[:, attr_idx] = (all_probs[:, attr_idx] >= thr).float()

        # Compute mAP with optimal thresholds (via AP again)
        aps_opt = torch.zeros(all_labels.shape[1])
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
            aps_opt[attr_idx] = ap
        mean_ap_opt = aps_opt[aps_opt > 0].mean().item() if len(aps_opt) > 0 else 0

        logger.info(f"** Epoch {epoch_num} ---> Validation mAP (thr=0.5): {100*mean_ap:.2f}, "
                    f"mAP (opt thr): {100*mean_ap_opt:.2f} **")

        if ckpt is not None:
            ckpt.save_best(acc=100 * mean_ap_opt, epoch=epoch_num, net=model)

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



