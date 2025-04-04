import sys
import os
import torch
import numpy as np
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


class color():
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

palate=[
        (0,0,0),
        (51,153,255),
        (255,0,0),
]
palate = np.array(palate,dtype=np.float32)/255.0

labels=['Background','Lung','Infection']

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    # bar = '█' * filled_length + ' ' * (bar_length - filled_length)
    bar = '■' * filled_length + '□' * (bar_length - filled_length)

    sys.stdout.write('\r%s |\033[34m%s\033[0m| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def hd95(masks, preds, num_class):
    NaN = np.nan
    masks = masks.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    metric_list = []
    for i in range(1,num_class):
        if np.sum(masks==i)>0 and np.sum(preds==i)>0:
            metric = medpy.metric.binary.hd95(result=(preds==i), reference=(masks==i))
        elif np.sum(masks==i)==0 and np.sum(preds==i)==0:
            metric = NaN
        else:
            metric = 0.0
        metric_list.append(metric)
    metric_list = np.array(metric_list)
    result = np.nanmean(metric_list)
    return result

class Save_Checkpoint(object):
    def __init__(self,filename):

        self.best_acc = 0.0
        self.best_epoch = 0
        self.folder = 'checkpoint'
        self.filename=filename
        self.best_path = '/content/drive/MyDrive/checkpoint/' + self.filename + '_best.pth'
        os.makedirs(self.folder, exist_ok=True)

    def save_best(self, acc, epoch, net):
        if self.best_acc < acc:
            print(color.BOLD+color.RED+'Saving best checkpoint...'+color.END)
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'best_epoch': epoch
            }
            self.best_epoch = epoch
            self.best_acc = acc
            torch.save(state, self.best_path)
        
    def best_accuracy(self):
        return self.best_acc
               
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



