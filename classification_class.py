from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import torch.nn as nn
import torch.nn.functional as F
import os
import torch

class ImagesDataset(Dataset):

    def __init__(self, annotations_file, img_dir,transform=None, target_transform=None):

        self.img_labels = pd.read_csv(img_dir + annotations_file,header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #reads image - uses io.imread because its a tif, also transposes to proper order for conversion to tensor
        image = io.imread(img_path).transpose(1,2,0)
        label = torch.tensor(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*122*122, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))   
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                   
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 16*122*122)
        output = self.fc1(output)

        return output