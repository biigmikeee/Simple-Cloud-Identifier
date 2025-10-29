import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import tranforms
from PIL import Image
import os

class CloudDataset(Dataset):

    #dataset for loading images
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    #cloud type classes
        self.classes = sorted(['cirriform', 'clear_sky', 'cumulonimbus', 'cumulus', 'high_cumuliform', 'stratiform', 'stratocumulus'])
        #mapping classes to indexes
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

        #loop for loading image paths and labels
        self.images = [] #holds directory of image and type
        self.labels = [] #holds index of cloud type

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)

            #getting all images in given class folder - put in arrays of paths and labels
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    



