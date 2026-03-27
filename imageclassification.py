import torch
import pandas as pd
import matplotlib.pyplot as plt
import classification_class
from torchvision.transforms import ToTensor,Compose,Normalize
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import time


#1 set up, train and test basic model based on z projected image
#####1.1 include defining size of images

#2 look into more confocal specific models, including using 

#3 how to choose the cnn architecture?

#3 tensors on cpu or gpu?


# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
    return(accuracy)

def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
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

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

        print("--- %s seconds ---" % (time.time() - start_time))



#test=ToTensor()(images[46]['image'].transpose(1,2,0))
#print (test.size())
transform_norm = Compose([ToTensor(),Normalize((0.5,0.5),(0.5,0.5))])
#target_transform = Compose([torch.tensor()])
train_images = classification_class.ImagesDataset(annotations_file='images.csv', img_dir='samples/',transform=transform_norm)
test_images = classification_class.ImagesDataset(annotations_file='test_images.csv', img_dir='test/',transform=transform_norm)



train_loader=DataLoader(train_images, batch_size=5, shuffle=True, num_workers=0)
test_loader=DataLoader(train_images, batch_size=5, shuffle=False, num_workers=0)

model = classification_class.Network()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

train(10)
print('Finished Training')

""" image,label = train_images[41]
fig = plt.figure()
plt.imshow(image.numpy()[1])
plt.show() """

