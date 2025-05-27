import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import CustomDataset

def get_all_training_set(data_paths):
    from transforms import get_train_transforms
    train_transforms=get_train_transforms()
    names= os.listdir(data_paths)
    all_datasets=[]
    for name in names:
        all_datasets.append(
            CustomDataset(os.path.join(data_paths,name,'training'),train_transforms)
        )
    return all_datasets


def get_all_test_set(data_paths):
    from transforms import get_test_transforms
    test_transforms=get_test_transforms()
    names= os.listdir(data_paths)
    all_datasets=[]
    for name in names:
        all_datasets.append(
            CustomDataset(os.path.join(data_paths,name,'test'),test_transforms)
        )
    return all_datasets

    
    


