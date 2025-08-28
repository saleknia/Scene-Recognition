import sys
# sys.path.append("/content/Scene-Recognition/model") 
# Instaling Libraries
import os
import copy
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
import argparse
from torch.backends import cudnn

import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import random_split
import torch.optim as optim
import scipy.io as sio

from model.Mobile_netV2 import Mobile_netV2
from model.Mobile_netV2_loss import Mobile_netV2_loss

from model.ResNet import ResNet
from model.ConvNext import ConvNext
from model.Combine import Combine
from model.Hybrid import Hybrid
from model.seg import seg
from model.DINOV3 import DINOV3
from model.DINOV2_att import DINOV2_att
import utils
from utils import color
from utils import Save_Checkpoint_accuracy
from trainer import trainer_func
from tester import tester_func
from dataset import superclasses, Fine_Grained_Dataset, Coarse_Grained_Dataset, SUN_717
from config import *
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

def main(args):

    # LOAD_DATA

    if TASK_NAME=='MIT-67' or TASK_NAME=='ImageNet':

        # Create a mapping: {original_class -> superclass_index}
        class_to_super = {}
        for idx, superclass in enumerate(superclasses):
            for cls in superclass:
                class_to_super[cls] = idx

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=384, scale=(0.2, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_valid = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if COARSE_GRAINED:
            trainset     = Coarse_Grained_Dataset('/content/MIT-67/train', class_to_super=class_to_super,transform=transform_train)
            validset     = Coarse_Grained_Dataset('/content/MIT-67/valid', class_to_super=class_to_super,transform=transform_test)
            testset      = Coarse_Grained_Dataset('/content/MIT-67/test' , class_to_super=class_to_super,transform=transform_test)
        
        elif FINE_GRAINED:

            trainset     = torchvision.datasets.ImageFolder(root='/content/MIT-67/train/', transform=transform_train)
            validset     = torchvision.datasets.ImageFolder(root='/content/MIT-67/valid/', transform=transform_test)
            testset      = torchvision.datasets.ImageFolder(root='/content/MIT-67/test/' , transform=transform_test)

            trainset     = Fine_Grained_Dataset(trainset, superclasses[SUPER_CLASS_INDEX-1])
            validset     = Fine_Grained_Dataset(validset, superclasses[SUPER_CLASS_INDEX-1])
            testset      = Fine_Grained_Dataset(testset , superclasses[SUPER_CLASS_INDEX-1])

        else:
            trainset     = torchvision.datasets.ImageFolder(root='/content/MIT-67/train/', transform=transform_train)
            validset     = torchvision.datasets.ImageFolder(root='/content/MIT-67/valid/', transform=transform_test)
            testset      = torchvision.datasets.ImageFolder(root='/content/MIT-67/test/' , transform=transform_test)       


        train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True , num_workers=NUM_WORKERS)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size = BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader  = torch.utils.data.DataLoader(testset , batch_size = 1         , shuffle=False, num_workers=NUM_WORKERS)

        if TASK_NAME=='MIT-67':
            NUM_CLASS = len(trainset.classes)
        else:
            NUM_CLASS = 512

        data_loader  = {'train':train_loader,'valid':valid_loader, 'test':test_loader}

    elif TASK_NAME=='SUN_717':

        # Load .mat files
        mat_data_images = sio.loadmat("/content/SUNAttributeDB/images.mat")
        mat_data_labels = sio.loadmat("/content/SUNAttributeDB/attributeLabels_continuous.mat")

        # Extract arrays
        image_paths = mat_data_images["images"]   # shape (14340, 1), each entry is array(['path'], dtype='<U..')
        labels = mat_data_labels["labels_cv"]     # shape (14340, 102)

        # Convert image_paths to a simple list of strings
        image_paths = [p[0][0] for p in image_paths]  # flatten
        image_paths = np.array([str(p) for p in image_paths])

        perm = np.random.permutation(len(image_paths))

        # apply the same shuffle to both arrays
        image_paths = image_paths[perm]
        labels      = labels[perm]

        transform_train = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Root directory where SUN images are extracted
        root_dir = "/content/images"
        # Create dataset
        trainset = SUN_717(image_paths[0:7170], root_dir, transform=transform_train)
        validset = SUN_717(image_paths[7170:] , root_dir, transform=transform_test)

        # Create dataloader
        train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True , num_workers=NUM_WORKERS)
        valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        NUM_CLASS = trainset.classes

        data_loader  = {'train':train_loader, 'test':valid_loader}

    elif TASK_NAME=='Scene-15':

        transform_train = transforms.Compose([
            # transforms.Resize((256, 256)),
            # transforms.CenterCrop(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=1.0),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = torchvision.datasets.ImageFolder(root='/content/Scene-15/train/', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        testset = torchvision.datasets.ImageFolder(root='/content/Scene-15/test/', transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset  , batch_size = 1         , shuffle=True, num_workers=NUM_WORKERS)

        data_loader={'train':train_loader,'valid':test_loader}
        
    elif TASK_NAME=='Standford40':

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # transforms.RandomErasing(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.ImageFolder(root='/content/StanfordActionDataset/train/',
                                        transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        testset = torchvision.datasets.ImageFolder(root='/content/StanfordActionDataset/test/', transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size =  1, shuffle=True, num_workers=NUM_WORKERS)

        NUM_CLASS = len(trainset.classes)

        data_loader={'train':train_loader,'valid':test_loader}

    elif TASK_NAME=='BU101+':

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.ImageFolder(root='/content/BU101/train/', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        testset = torchvision.datasets.ImageFolder(root='/content/BU101/test/', transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        data_loader={'train':train_loader,'valid':test_loader}

    # MODEL_INITIALIZE

    if MODEL_NAME == 'Mobile_NetV2':
        model = Mobile_netV2(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'Mobile_NetV2_loss':
        model = Mobile_netV2_loss(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'ResNet':
        model = ResNet(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'ConvNext':
        model = ConvNext(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'Combine':
        model = Combine(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'Hybrid':
        model = Hybrid(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'seg':
        model = seg(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'DINOV3':
        model = DINOV3(num_classes=NUM_CLASS).to(DEVICE)

    elif MODEL_NAME == 'DINOV2_att':
        model = DINOV2_att(num_classes=NUM_CLASS).to(DEVICE)
        
    else: 
        raise TypeError('Please enter a valid name for the model type')

    # LOAD_MODEL

    num_parameters = utils.count_parameters(model)

    model_table = tabulate(
        tabular_data=[[MODEL_NAME, f'{num_parameters:.2f} M', DEVICE]],
        headers=['Builded Model', '#Parameters', 'Device'],
        tablefmt="fancy_grid"
        )
    logger.info(model_table)

    if SAVE_MODEL:
        checkpoint = Save_Checkpoint_accuracy(CKPT_NAME)
    else:
        checkpoint = None

    
    checkpoint_path = '/content/drive/MyDrive/checkpoint/'+CKPT_NAME+'_best.pth'  

    if LOAD_MODEL:
        logger.info('Loading Checkpoint...')
        if os.path.isfile(checkpoint_path):
            pretrained_model_path = checkpoint_path
            loaded_data = torch.load(pretrained_model_path, map_location='cuda')
            pretrained = loaded_data['net']
            model2_dict = model.state_dict()
            state_dict = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}
            model2_dict.update(state_dict)
            model.load_state_dict(model2_dict)
        else:
            logger.info(f'No Such file : {checkpoint_path}')
        logger.info('\n')

    if args.train=='True':
        #######################################################################################################################################
        optimizer      = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)   
        # optimizer      = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        total_batchs   = len(data_loader['train'])
        max_iterations = NUM_EPOCHS * total_batchs
        #######################################################################################################################################

        if POLY_LR is True:
            lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=max_iterations, power=0.9)
        else:
            lr_scheduler =  None  

    if args.train=='True':
        logger.info(50*'*')
        logger.info('Training Phase')
        logger.info(50*'*')
        for epoch in range(1, NUM_EPOCHS+1):
            trainer_func(
                epoch_num=epoch,
                model=model,
                dataloader=data_loader,
                optimizer=optimizer,
                device=DEVICE,
                ckpt=checkpoint,                
                num_class=NUM_CLASS,
                lr_scheduler=lr_scheduler,
                logger=logger)

    if (args.inference=='True') and (os.path.isfile(checkpoint_path)):
        loaded_data = torch.load(checkpoint_path, map_location='cuda')
        pretrained  = loaded_data['net']
        model2_dict = model.state_dict()
        state_dict  = {k:v for k,v in pretrained.items() if ((k in model2_dict.keys()) and (v.shape==model2_dict[k].shape))}

        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)

        acc=loaded_data['acc']
        best_epoch=loaded_data['best_epoch']

        logger.info(50*'*')
        logger.info(f'Best Acc Over Validation Set: {acc:.2f}')
        logger.info(f'Best Epoch: {best_epoch}')

        logger.info(50*'*')
        logger.info('Inference Phase')
        logger.info(50*'*')
        tester_func(
            model=copy.deepcopy(model),
            dataloader=data_loader,
            device=DEVICE,
            ckpt=None,
            num_class=NUM_CLASS,
            logger=logger)
    else:
        logger.info(50*'*')
        logger.info('Inference Phase')
        logger.info(50*'*')
        tester_func(
            model=copy.deepcopy(model),
            dataloader=data_loader,
            device=DEVICE,
            ckpt=None,
            num_class=NUM_CLASS,
            logger=logger)

    logger.info(50*'*')
    logger.info('\n')

parser = argparse.ArgumentParser()
parser.add_argument('--inference', type=str, default='True')
parser.add_argument('--train'    , type=str, default='True')
parser.add_argument('--KF'       , type=str, default='False')
parser.add_argument('--fold'     , type=str, default='0')

args = parser.parse_args()

def worker_init(worker_id):
    random.seed(SEED + worker_id)

if __name__ == "__main__":
    
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
    
    random.seed(SEED)    
    np.random.seed(SEED)  
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) 

    if args.KF=='True':
        fold = int(args.fold)

    main(args)
    
    # if args.KF=='True':
    #     os.system(f'mv /content/drive/MyDrive/checkpoint/{CKPT_NAME}_best.pth /content/drive/MyDrive/checkpoint/{CKPT_NAME}_best_fold_{fold}.pth')
    