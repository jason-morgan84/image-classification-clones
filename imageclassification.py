import torch
#import pandas as pd
import matplotlib.pyplot as plt
import classification_class
from torchvision.transforms import ToTensor,Compose,Normalize
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import time
import numpy as np


#0: look into loss, training loss v validation loss
#1: add section to testAccuracy to return accuracy by group


#2 look into more confocal specific models

#3 how to choose the cnn architecture?


batch_size=5
classes = ("AR","ARS")

def testImage(index):
    model.eval()

    image, label = test_images[index]
    print(label)

# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader
    model.eval()
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imagegrid = torchvision.utils.make_grid(images)
    img = torchvision.transforms.ToPILImage()(imagegrid[0])
    img.show()

    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs.data, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(batch_size)))


# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images.to(device))
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy

def train(num_epochs):
    model.train()
    best_accuracy = 0.0
    
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()


            # print loss statistics twice per epoch
            running_loss += loss.item()     # extract the loss value
            if i % int((n_batches - 1)/2) == int((n_batches - 1)/2 - 1):    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / (n_batches)))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % accuracy,"(",round((time.time() - start_time),3),"seconds)")
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

        #print("--- %s seconds ---" % )

# Define your execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("The model will be running on", device, "device")

transform_norm = Compose([ToTensor(),Normalize((0.5,0.5),(0.5,0.5))])


train_images = classification_class.ImagesDataset(annotations_file='images.csv', img_dir='samples/',transform=transform_norm)
n_train_images = len(train_images)
n_batches = n_train_images/batch_size

test_images = classification_class.ImagesDataset(annotations_file='test_images.csv', img_dir='test/',transform=transform_norm)



train_loader=DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader=DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=0)

model = classification_class.Network()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Let's build our model
train(5)
print('Finished Training')

# Let's load the model we just created and test the accuracy per label
model = classification_class.Network()
path = "myFirstModel.pth"
model.load_state_dict(torch.load(path))
print('Loaded model')

# Test with batch of images
testBatch()



