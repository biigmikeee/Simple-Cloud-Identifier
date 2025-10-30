import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from collections import Counter

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

    #dataset length method for PyTorch Dataloader
    def __len__(self):
        return len(self.images)
    
    #get an image and its label, for training
    #should return an image and its label
    def __getitem__(self, idx):

        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB') #-> convert to RBG to have consistent color channels for all images
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    #calculating class weights to deal with the underrepresentation of data and imbalances - with this the model learns that error on underrepresented classes are costly. forces more attention on less frequent classes.
    def get_class_weights(self):

        label_counts = Counter(self.labels)
        total = len(self.labels)

        num_classes = len(self.classes)
        weights = []

        for class_idx in range(num_classes):
            count = label_counts[class_idx]
            weight = total / (num_classes * count)
            weights.append(weight)

        return torch.FloatTensor(weights) #for loss function in pytorch we want a floattensor
    
    #training transforms for altering training data so model can see different variations -> better generalization. Also increases dataset size
    # testing -> no transformation just normalization   
    #for better performance on varying cloud images
    def get_transforms(train=True):

        if train:

            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


