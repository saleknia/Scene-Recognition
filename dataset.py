import numpy as np
import torch

# Your superclass arrays
superclass_1 = np.array(['airport_inside', 'auditorium', 'bar', 'bowling', 'casino',
       'church_inside', 'cloister', 'concert_hall', 'corridor',
       'elevator', 'fastfood_restaurant', 'inside_bus', 'inside_subway',
       'laundromat', 'lobby', 'mall', 'movietheater', 'poolinside',
       'restaurant', 'restaurant_kitchen', 'stairscase', 'subway',
       'trainstation'], dtype='<U19')

superclass_2 = np.array(['artstudio', 'bakery', 'bookstore', 'buffet', 'closet',
       'clothingstore', 'deli', 'dining_room', 'florist', 'garage',
       'greenhouse', 'grocerystore', 'jewelleryshop', 'library',
       'livingroom', 'museum', 'pantry', 'shoeshop', 'toystore',
       'videostore', 'warehouse', 'winecellar'], dtype='<U19')

superclass_3 = np.array(['bathroom', 'bedroom', 'children_room', 'classroom',
       'computerroom', 'dentaloffice', 'gameroom', 'gym', 'hairsalon',
       'hospitalroom', 'kindergarden', 'kitchen', 'laboratorywet',
       'locker_room', 'meeting_room', 'nursery', 'office',
       'operating_room', 'prisoncell', 'studiomusic', 'tv_studio',
       'waitingroom'], dtype='<U19')

superclasses = [superclass_1, superclass_2, superclass_3]

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Custom Dataset class
class Coarse_Grained_Dataset(Dataset):
    def __init__(self, root, class_to_super, transform):
        self.original_dataset = ImageFolder(root, transform=transform)
        self.class_to_super   = class_to_super
        self.super_labels = [
            self.class_to_super[self.original_dataset.classes[orig_label]]
            for _, orig_label in self.original_dataset.imgs
        ]
        self.classes = torch.unique(self.super_labels)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, _ = self.original_dataset[idx]  # Ignore original label
        super_label = self.super_labels[idx]
        return image, super_label  # Now returns (image, superclass_label)

class Fine_Grained_Dataset(Dataset):
    def __init__(self, original_dataset, superclass_classes):
        self.original_dataset = original_dataset
        self.superclass_classes = superclass_classes
        
        # Create a mapping from original class names to new labels 
        self.class_to_new_label = {
            cls_name: idx for idx, cls_name in enumerate(superclass_classes)
        }
        
        # Filter indices of samples belonging to Superclass 
        self.filtered_indices = [
            i for i, (_, label) in enumerate(original_dataset.imgs)
            if original_dataset.classes[label] in superclass_classes
        ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # Get the original sample (image, original_label)
        original_idx = self.filtered_indices[idx]
        image, orig_label = self.original_dataset[original_idx]
        
        # Get the new label
        orig_class_name = self.original_dataset.classes[orig_label]
        new_label = self.class_to_new_label[orig_class_name]
        
        return image, new_label