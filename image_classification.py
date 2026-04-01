import torch
import classification_class
from torchvision.transforms import Compose,Normalize
from torch.utils.data import DataLoader

import model_test
import model_train
import model_interpret

#-1: what does captum output mean?
#0: look into loss, training loss v validation loss
#1: Fix artifacts in outputted images?
#2 look into more confocal specific models
#3 how to choose the cnn architecture?

batch_size = 5
classes = ("AR","ARS")

# Define your execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

print("The model will be running on", device, "device")

transform_norm = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_images = classification_class.ImagesDataset(annotations_file='images.csv', img_dir='samples/',transform=transform_norm)
#n_train_images = len(train_images)
n_batches = len(train_images)/batch_size

test_images = classification_class.ImagesDataset(annotations_file='test_images.csv', img_dir='test/',transform=transform_norm)

train_loader=DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader=DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=0)

model = classification_class.Network()

# Let's build our model
#model_train.train(model, device, train_loader, test_loader, 10)
print('Finished Training')


# Let's load the model we just created and test the accuracy per label
path = "myFirstModel.pth"
model.load_state_dict(torch.load(path))
print('Loaded model')

#test_class_accuracy(classes)

# Test with batch of images
#test_batch()

model_interpret.model_occlusion(model,test_images)



