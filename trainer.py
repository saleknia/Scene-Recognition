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

		return loss

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
		loss = (self.w_dist * self.rkd_dist(feat_s, feat_t) + self.w_angle * self.rkd_angle(feat_s, feat_t))
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
    # torch.autograd.set_detect_anomaly(True)
    print(f'Epoch: {epoch_num} ---> Train , lr: {optimizer.param_groups[0]["lr"]}')
    
    model = model.to('cuda')
    model.train()

    loss_ce_total = utils.AverageMeter()
    loss_di_total = utils.AverageMeter() 
    loss_total    = utils.AverageMeter() 

    # accuracy = utils.AverageMeter()
    metric = MulticlassAccuracy(average="macro", num_classes=num_class).to('cuda')
    # accuracy = mAPMeter()

    loss_ce = CrossEntropyLoss(label_smoothing=0.0)
    loss_di = SP()

    total_batchs = len(dataloader['train'])
    loader       = dataloader['train'] 

    base_iter    = (epoch_num-1) * total_batchs
    iter_num     = base_iter

    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)

        targets = targets.float()
        outputs = model(inputs)

        if type(outputs)==tuple:
            predictions = torch.argmax(input=torch.softmax(outputs[0], dim=1),dim=1).long()
            # outputs, aux = outputs[0], outputs[1]
            # goals = torch.tensor([mapping[x] for x in targets.long()]).long().cuda()
            # loss  = loss_ce(outputs, targets.long()) + (loss_ce(aux, goals.long()) * 0.5)
            T       = 2.0
            ce_loss = loss_ce(outputs[0], targets.long())
            di_loss = F.kl_div(F.log_softmax(outputs[0]/T, dim=1), F.softmax(outputs[1]/T, dim=1), reduction='batchmean') * T * T
            loss    = ce_loss + di_loss
        else:
            predictions = torch.argmax(input=torch.softmax(outputs, dim=1),dim=1).long()
            ce_loss = loss_ce(outputs, targets.long())
            di_loss = 0.0 
            loss    = ce_loss + di_loss

        loss_ce_total.update(ce_loss)
        loss_di_total.update(di_loss)
        loss_total.update(loss)

        metric.update(predictions, targets.long())

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step() 

        print_progress(
            iteration=batch_idx+1,
            total=total_batchs,
            prefix=f'Train {epoch_num} Batch {batch_idx+1}/{total_batchs} ',
            # suffix=f'CE_Loss = {loss_ce_total.avg:.4f} , Accuracy = {100 * metric.compute():.4f}',   
            suffix=f'CE_loss = {loss_ce_total.avg:.4f} , distillation_loss = {loss_di_total.avg:.4f} , Accuracy = {100 * metric.compute():.4f}',                 
            # suffix=f'CE_loss = {loss_ce_total.avg:.4f} , disparity_loss = {loss_disparity_total.avg:.4f} , Accuracy = {100 * accuracy.avg:.4f}',   
            bar_length=45
        )  
  
    Acc = 100 * metric.compute()
        
    logger.info(f'Epoch: {epoch_num} ---> Train , Loss = {loss_ce_total.avg:.4f}, Accuracy = {Acc:.2f} , lr = {optimizer.param_groups[0]["lr"]}')
    # valid_func(epoch_num,copy.deepcopy(model),dataloader,device,ckpt,num_class,logger)
    if ckpt is not None:
       ckpt.save_best(acc=Acc, epoch=epoch_num, net=model)

