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

class SP(nn.Module):
	'''
	Similarity-Preserving Knowledge Distillation
	https://arxiv.org/pdf/1907.09682.pdf
	'''
	def __init__(self):
		super(SP, self).__init__()

	def forward(self, fm_s, fm_t):
		fm_s = fm_s.view(fm_s.size(0), -1)
		G_s  = torch.mm(fm_s, fm_s.t())
		norm_G_s = F.normalize(G_s, p=2, dim=1)

		fm_t = fm_t.view(fm_t.size(0), -1)
		G_t  = torch.mm(fm_t, fm_t.t())
		norm_G_t = F.normalize(G_t, p=2, dim=1)

		loss = F.mse_loss(norm_G_s, norm_G_t)

		return loss * 1000.0

class RKD(nn.Module):
	'''
	Relational Knowledge Distillation
	https://arxiv.org/pdf/1904.05068.pdf
	'''
	def __init__(self, w_dist=50, w_angle=100):
		super(RKD, self).__init__()

		self.w_dist  = 50
		self.w_angle = 100

	def forward(self, feat_s, feat_t):
		loss = (self.w_dist * self.rkd_dist(feat_s, feat_t) + self.w_angle * self.rkd_angle(feat_s, feat_t)) / 2.0
		return loss

	def rkd_dist(self, feat_s, feat_t):
		feat_t_dist = self.pdist(feat_t, squared=False)
		mean_feat_t_dist = feat_t_dist[feat_t_dist>0].mean()
		feat_t_dist = feat_t_dist / mean_feat_t_dist

		feat_s_dist = self.pdist(feat_s, squared=False)
		mean_feat_s_dist = feat_s_dist[feat_s_dist>0].mean()
		feat_s_dist = feat_s_dist / mean_feat_s_dist

		loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

		return loss

	def rkd_angle(self, feat_s, feat_t):
		# N x C --> N x N x C
		feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
		norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
		feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)

		feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
		norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
		feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)

		loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

		return loss

	def pdist(self, feat, squared=False, eps=1e-12):
		feat_square = feat.pow(2).sum(dim=1)
		feat_prod   = torch.mm(feat, feat.t())
		feat_dist   = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

		if not squared:
			feat_dist = feat_dist.sqrt()

		feat_dist = feat_dist.clone()
		feat_dist[range(len(feat)), range(len(feat))] = 0

		return feat_dist

mapping = [0, 1, 0, 1, 0, 2, 2, 1, 0, 1, 0, 2, 0, 2, 0, 1, 1, 2, 0, 0, 1, 2,
       1, 0, 0, 1, 2, 1, 1, 1, 2, 2, 2, 0, 0, 1, 2, 2, 2, 0, 1, 1, 0, 2,
       0, 2, 0, 1, 2, 2, 2, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 1, 2, 1,
       1]

warnings.filterwarnings("ignore")

from sklearn.metrics import average_precision_score
import torch

def trainer_func(epoch_num, model, dataloader, optimizer, device, ckpt, num_class, lr_scheduler, logger):
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    
    model = model.to(device)
    model.train()

    loss_total     = utils.AverageMeter() 
    soft_acc_total = utils.AverageMeter()
    map_total      = utils.AverageMeter()  # running mAP

    loss_bce = nn.BCEWithLogitsLoss()

    total_batchs = len(dataloader['train'])
    loader       = dataloader['train'] 

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)

        # ---- Binarize labels ----
        outputs = model(inputs)

        binary_targets = (targets >= 0.6).float()

        mask = ((targets >= 0.6) | (targets <= 0.1)).float()
        num_valid_labels = mask.sum() # Total number of valid (non-ignored) labels in this batch

        # ---- Calculate Masked BCE Loss ----
        loss_per_element = loss_bce(outputs, binary_targets)
        masked_loss = loss_per_element * mask
        loss = masked_loss.sum() / torch.clamp(num_valid_labels, min=1.0) 

        loss_total.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step() 

        # ---- Soft Accuracy & batch mAP ----
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            # Soft accuracy
            correct = (preds == targets).float().mean().item()
            soft_acc_total.update(correct, n=inputs.size(0))

            # Batch-wise mAP
            probs_np   = probs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            ap_per_class = []
            for i in range(targets_np.shape[1]):
                if targets_np[:, i].sum() == 0:
                    continue  # skip class if no positives in this batch
                ap = average_precision_score(targets_np[:, i], probs_np[:, i])
                ap_per_class.append(ap)
            batch_map = sum(ap_per_class) / len(ap_per_class) if ap_per_class else 0
            map_total.update(batch_map, n=inputs.size(0))

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            suffix=f'CE_Loss = {loss_total.avg:.4f}, SoftAcc = {100 * soft_acc_total.avg:.2f}, mAP = {100 * map_total.avg:.2f}',   
            bar_length=45
        )  

    # ---- Log at end of epoch ----
    logger.info(
        f'Epoch: {epoch_num} ---> Train , Loss = {loss_total.avg:.4f}, '
        f'SoftAcc = {100 * soft_acc_total.avg:.2f}, mAP = {100*map_total.avg:.2f}, '
        f'lr = {optimizer.param_groups[0]["lr"]}'
    )

    if ckpt is not None:
        ckpt.save_best(loss=loss_total.avg, epoch=epoch_num, net=model)



