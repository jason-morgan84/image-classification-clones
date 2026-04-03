from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from torchvision.io import read_image

class ImagesDataset(Dataset):

    def __init__(self, annotations_file, img_dir,transform=None, target_transform=None):

        self.img_labels = pd.read_csv(img_dir + annotations_file,header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_name = self.img_labels.iloc[idx,0]
        img_path = os.path.join(self.img_dir, image_name)
        #reads image - uses io.imread because its a tif, also transposes to proper order for conversion to tensor
        image = read_image(img_path).float()/255
        label = torch.tensor(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, image_name
    
    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*122*122, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))   
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                   
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*122*122)
        output = self.fc1(output)

        return output