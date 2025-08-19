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

def trainer_func(epoch_num,model,dataloader,optimizer,device,ckpt,num_class,lr_scheduler,logger):
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    
    model = model.to(device)
    model.train()

    loss_total     = utils.AverageMeter() 
    soft_acc_total = utils.AverageMeter()

    loss_bce = nn.BCEWithLogitsLoss()

    total_batchs = len(dataloader['train'])
    loader       = dataloader['train'] 

    base_iter    = (epoch_num-1) * total_batchs
    iter_num     = base_iter

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        loss = loss_bce(outputs, targets)

        loss_total.update(loss)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step() 

        # ---- Soft Accuracy ----
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            eps = 0.1
            
            mask = targets > 0  # only nonzero labels matter
            correct = ((torch.abs(probs - targets) < eps) & mask).float().sum()
            total = mask.float().sum()

            if total > 0:
                soft_acc = (correct / total).item()
            else:
                soft_acc = 1.0  # or skip this batch if no nonzero labels

            # ---- Log at end of epoch ----
            logger.info(
                f'Epoch: {epoch_num} ---> Train , Loss = {loss_total.avg:.4f}, '
                f'SoftAcc = {100 * soft_acc_total.avg:.2f}, lr = {optimizer.param_groups[0]["lr"]}'
            )

    if ckpt is not None:
        ckpt.save_best(loss=loss_total.avg, epoch=epoch_num, net=model)

