from torchvision import datasets
from config import *
from transforms import *
from torch.utils.data import DataLoader

# if isinstance(GPU,list):
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os



def get_datasets(pretrained):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        TRAIN_DIR, 
        transform=(get_train_transform(IMAGE_DIMS, pretrained))
    )
    dataset_valid = datasets.ImageFolder(
        VAL_DIR, 
        transform=(get_valid_transform(IMAGE_DIMS, pretrained))
    )
    # dataset_size = len(dataset)
    # # Calculate the validation dataset size.
    # valid_size = int(VALID_SPLIT*dataset_size)
    # # Radomize the data indices.
    # indices = torch.randperm(len(dataset)).tolist()
    # # Training and validation sets.
    # dataset_train = Subset(dataset, indices[:-valid_size])
    # dataset_valid = Subset(dataset_test, indices[-valid_size:])
    
    return dataset_train, dataset_valid, dataset_train.classes



from torchsampler import ImbalancedDatasetSampler
def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, 
        
        batch_size=BATCH_SIZE, 
        # sampler=DistributedSampler(dataset_train),
        shuffle=False, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid,
        # sampler=DistributedSampler(dataset_valid),
        batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader 








# from torchsampler import ImbalancedDatasetSampler
# def get_datasets(pretrained):
#     """
#     Function to prepare the Datasets.
#     :param pretrained: Boolean, True or False.
#     Returns the training and validation datasets along 
#     with the class names.
#     """
#     dataset = datasets.ImageFolder(
#         ROOT_DIR, 
#         transform=(get_train_transform(IMAGE_DIMS, pretrained))
#     )
#     dataset_test = datasets.ImageFolder(
#         ROOT_DIR, 
#         transform=(get_valid_transform(IMAGE_DIMS, pretrained))
#     )
#     dataset_size = len(dataset)
#     # Calculate the validation dataset size.
#     valid_size = int(VALID_SPLIT*dataset_size)
#     # Radomize the data indices.
#     indices = torch.randperm(len(dataset)).tolist()
#     # Training and validation sets.
#     dataset_train = Subset(dataset, indices[:-valid_size])
#     dataset_valid = Subset(dataset_test, indices[-valid_size:])
    
#     return dataset_train, dataset_valid, dataset.classes
# def get_data_loaders(dataset_train, dataset_valid):
#     """
#     Prepares the training and validation data loaders.
#     :param dataset_train: The training dataset.
#     :param dataset_valid: The validation dataset.
#     Returns the training and validation data loaders.
#     """
#     train_loader = DataLoader(
#         dataset_train, 
        
#         batch_size=BATCH_SIZE, 
#         sampler=ImbalancedDatasetSampler(dataset_train),
#         shuffle=False, num_workers=NUM_WORKERS
#     )
#     valid_loader = DataLoader(
#         dataset_valid,
#         sampler=ImbalancedDatasetSampler(dataset_valid),
#         batch_size=BATCH_SIZE, 
#         shuffle=False, num_workers=NUM_WORKERS
#     )
#     return train_loader, valid_loader 
