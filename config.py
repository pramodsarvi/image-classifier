ROOT_DIR='/home/pramod/Documents/archive'
GPU = 0
GPU = "0"
"""
world_size
"""
# world_size = len(GPU)
# temp = []
# if isinstance(GPU,list):
#     for i in range(len(GPU)):
#         temp.append("cuda:"+str(GPU[i]))
# GPU = temp
# GPU = [0,1,2,3,4,5] for multi-GPU
TRAIN_DIR = ROOT_DIR+'/train'
VAL_DIR = ROOT_DIR+'/val'
SAVE_PATH = ROOT_DIR+ '/result'
BATCH_SIZE = 2
NUM_WORKERS = 4
VALID_SPLIT=0.2
SAVE_EPOCH=15 
IMAGE_DIMS=[224,224]    
EPOCHS=10
GRID_SIZE=[3,2]
DEVICE = 'cuda:0'
lr = 0.001
QAT = False