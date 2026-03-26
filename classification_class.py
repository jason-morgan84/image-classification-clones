from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from skimage import io

class ImagesDataset(Dataset):

    def __init__(self, data_file, directory):

        self.image_data = pd.read_csv(directory + data_file,header=None)
        self.directory = directory

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = self.directory + self.image_data.iloc[index, 0]
        image = io.imread(img_name)
        type = self.image_data.iloc[index, 1]
        sample = {'image': image, 'type': type}
        return sample