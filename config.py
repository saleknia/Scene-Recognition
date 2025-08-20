import os
import torch
import torchvision
import logging
from utils import color
from tabulate import tabulate
import ml_collections

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

SEED = 42

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['PYTHONHASHSEED'] = str(SEED)

##########################################################################
# Log Directories
##########################################################################
tensorboard = False
tensorboard_folder = './logs/tensorboard'
log = True
logging_folder = './logs/logging'

if log:
    logging_log = logging_folder
    if not os.path.isdir(logging_log):
        os.makedirs(logging_log)
    logger = logger_config(log_path = logging_log + '/training_log.log')
    logger.info(f'Logging Directory: {logging_log}')   
##########################################################################

LEARNING_RATE = 0.001
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 64
NUM_EPOCHS    = 20
NUM_WORKERS   = 4
IMAGE_HEIGHT  = 224
IMAGE_WIDTH   = 224
PIN_MEMORY    = True

LOAD_MODEL = True
CONTINUE   = True

TEACHER    = False

SAVE_MODEL = True
POLY_LR    = True
DOWNLOAD   = False


task_ids = ['1','2','3']
task_table = tabulate(
                    tabular_data=[
                        ['NORMAL_TRAINING',1],
                        ['COARSE_GRAINED' ,2],
                        ['FINE_GRAINED'   ,3]],
                    headers=['Task Name', 'ID'],
                    tablefmt="fancy_grid"
                    )
print(task_table)
task_id  = input('Choose Training Setup IndexSetup Index: ')
assert (task_id in task_ids),'Setup Index is Incorrect!'
task_id = int(task_id)

if task_id == 1:
    NORMAL_TRAINING = True
    COARSE_GRAINED  = False
    FINE_GRAINED    = False

if task_id == 2:
    NORMAL_TRAINING = False
    COARSE_GRAINED  = True
    FINE_GRAINED    = False
    DESCRIPTION     = '_COARSE_GRAINED'
    
if task_id == 3:
    NORMAL_TRAINING = False
    COARSE_GRAINED  = False
    FINE_GRAINED    = True

    task_ids = ['1','2','3']
    task_table = tabulate(
                        tabular_data=[
                            ['Super_class_1' ,1],
                            ['Super_class_2' ,2],
                            ['Super_class_3' ,3]],
                        headers=['Task Name', 'ID'],
                        tablefmt="fancy_grid"
                        )

    print(task_table)
    task_id = input('Enter Your Super Class:  ')
    assert (task_id in task_ids),'Super Class Number is Incorrect!'
    
    DESCRIPTION       = '_FINE_GRAINED_' + task_id
    SUPER_CLASS_INDEX = int(task_id)


os.environ['PYTHONHASHSEED'] = str(SEED)

task_ids = ['1','2','3','4','5','6']
task_table = tabulate(
                    tabular_data=[
                        ['Standford40'  , 1],
                        ['BU101+'       , 2],
                        ['MIT-67'       , 3],
                        ['Scene-15'     , 4],
                        ['SUNAttribute' , 5],
                        ['ImageNet'     , 6]],
                    headers=['Task Name', 'ID'],
                    tablefmt="fancy_grid"
                    )

print(task_table)
task_id = input('Enter Task ID:  ')
assert (task_id in task_ids),'ID is Incorrect.'
task_id = int(task_id)

if task_id==1:
    NUM_CLASS = 40
    TASK_NAME = 'Standford40'

elif task_id==2:
    NUM_CLASS = 101
    TASK_NAME = 'BU101+'

elif task_id==3:
    NUM_CLASS = 67
    TASK_NAME = 'MIT-67'

elif task_id==4:
    NUM_CLASS = 15
    TASK_NAME = 'Scene-15'

elif task_id==5:
    NUM_CLASS = 102
    TASK_NAME = 'SUNAttribute'

elif task_id==6:
    NUM_CLASS = 2048
    TASK_NAME = 'ImageNet'

model_ids = ['1','2','3','4','5','6','7','8']
model_table = tabulate(
                    tabular_data=[
                        ['Mobile_netV2'     , 1],
                        ['Mobile_netV2_loss', 2],
                        ['ResNet'           , 3],
                        ['ConvNext'         , 4],
                        ['Combine'          , 5],
                        ['Hybrid'           , 6],
                        ['seg'              , 7],
                        ['DINOV3'           , 8]],
                    headers=['Model Name', 'ID'],
                    tablefmt="fancy_grid"
                    )

print(model_table)
model_id = input('Enter Model ID:  ')
assert (model_id in model_ids),'ID is Incorrect.'
model_id = int(model_id)

if model_id==1:
    MODEL_NAME = 'Mobile_NetV2'

elif model_id==2:
    MODEL_NAME = 'Mobile_NetV2_loss'

elif model_id==3:
    MODEL_NAME = 'ResNet'

elif model_id==4:
    MODEL_NAME = 'ConvNext'

elif model_id==5:
    MODEL_NAME = 'Combine'

elif model_id==6:
    MODEL_NAME = 'Hybrid'

elif model_id==7:
    MODEL_NAME = 'seg'

elif model_id==8:
    MODEL_NAME = 'DINOV3'

if NORMAL_TRAINING:
    CKPT_NAME = MODEL_NAME + '_' + TASK_NAME
else:
    CKPT_NAME = MODEL_NAME + '_' + TASK_NAME + DESCRIPTION

table = tabulate(
    tabular_data=[
        ['Learning Rate', LEARNING_RATE],
        ['Num Classes', NUM_CLASS],
        ['Device', DEVICE],
        ['Batch Size', BATCH_SIZE],
        ['POLY_LR', POLY_LR],
        ['Num Epochs', NUM_EPOCHS],
        ['Num Workers', NUM_WORKERS],
        ['Image Height', IMAGE_HEIGHT],
        ['Image Width', IMAGE_WIDTH],
        ['Pin Memory', PIN_MEMORY],
        ['Load Model', LOAD_MODEL],
        ['Save Model', SAVE_MODEL],
        ['Download Dataset', DOWNLOAD],
        ['Model Name', MODEL_NAME],
        ['Seed', SEED],
        ['Task Name', TASK_NAME],
        ['GPU', torch.cuda.get_device_name(0)],
        ['Torch', torch.__version__],
        ['Torchvision', torchvision.__version__],
        ['Checkpoint Name', CKPT_NAME]],
    headers=['Hyperparameter', 'Value'],
    tablefmt="fancy_grid"
    )

logger.info(table)







