import torch
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
import classification_class
from torchvision.transforms import ToTensor

#1 set up, train and test basic model based on z projected image
#####1.1 include defining size of images

#2 look into more confocal specific models, including using 

#3 tensors on cpu or gpu?

#image_data = pd.read_csv('samples/images.csv')
#n=5
#image_name = image_data.iloc[n,0]
#print("Name:",image_name)
#print("Type:",image_data.iloc[n,1])

images = classification_class.ImagesDataset(data_file='images.csv', directory='samples/')



"""for n, image in enumerate(images):
    print(n,image['image'].shape,image['type'])
    test = ToTensor()(image['image'].transpose(2,0,1))
    #print(type(test),test.size())
    if n>=0:
        break"""

test=ToTensor()(images[46]['image'].transpose(2,0,1))
print(images[46]['type'])


fig = plt.figure()
plt.imshow(test.numpy().transpose(2,0,1)[0])
plt.show()

