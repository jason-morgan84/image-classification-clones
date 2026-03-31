import torch
from sympy import true

import classification_class
from torchvision.transforms import Compose,Normalize
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import time
from captum.attr import Occlusion
import numpy as np




#-1: what does captum output mean?
#0: look into loss, training loss v validation loss
#1: Fix artifacts in outputted images?
#2 look into more confocal specific models
#3 how to choose the cnn architecture?


batch_size=5
classes = ("AR","ARS")

# Function to test the model with a batch of images and show the labels predictions
def test_batch():
    # get batch of images from the test DataLoader
    model.eval()
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imagegrid = torchvision.utils.make_grid(images)
    print(imagegrid.size())
    #imagegrid = imagegrid.transpose(1,2,0)
    img = torchvision.transforms.ToPILImage()(imagegrid)
    img.show()

    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    model.to(device)
    outputs = model(images.to(device))

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs.data, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(batch_size)))



# Function to test the model with the test dataset and print the accuracy for the test images
def test_accuracy():
    
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


def test_class_accuracy(groups):

    model.to(device)
    model.eval()
    n_groups = len(groups)
    group_total = [0] * n_groups
    group_accuracy = []
    total = [0] * n_groups

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            # run the model on the test set to predict labels
            outputs = model(images.to(device))
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            test = (predicted==labels.squeeze())
            for n, item in enumerate(labels):
                item_group = item.item()
                total[item_group] += 1
                if test[n].item()==true:
                    group_total[item_group] += 1

        for n,group in enumerate(classes):
            group_accuracy.append((group_total[n]/total[n])*100)
            print(group+" ("+str(n)+") accuracy: "+str(round(group_accuracy[n],2))+"%")


def train(num_epochs):
    model.train()

    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
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
                      (epoch + 1, i + 1, running_loss / n_batches))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = test_accuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % accuracy,"(",round((time.time() - start_time),3),"seconds)")
        
        # we want to save the model if the accuracy is the best
        """if accuracy > best_accuracy:
            save_model()
            best_accuracy = accuracy"""

    torch.save(model.state_dict(), "./myFirstModel.pth")
# Define your execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

print("The model will be running on", device, "device")

transform_norm = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_images = classification_class.ImagesDataset(annotations_file='images.csv', img_dir='samples/',transform=transform_norm)
n_train_images = len(train_images)
n_batches = n_train_images/batch_size

test_images = classification_class.ImagesDataset(annotations_file='test_images.csv', img_dir='test/',transform=transform_norm)

image,_=train_images[1]

train_loader=DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader=DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=0)

model = classification_class.Network()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


example,label= next(iter(train_loader))
# Let's build our model
#train(10)
#print('Finished Training')



# Let's load the model we just created and test the accuracy per label
path = "myFirstModel.pth"
model.load_state_dict(torch.load(path))
print('Loaded model')

#test_class_accuracy(classes)

# Test with batch of images
#test_batch()



model.eval()

occlusion = Occlusion(model)
strides = (3,9, 9)               # smaller = more fine-grained attribution but slower
target=0,                       # AR index
sliding_window_shapes=(3,15, 15)  # choose size enough to change object appearance
baselines = 0                     # values to occlude the image with 0 corresponds to gray

interpretation_loader=DataLoader(test_images, batch_size=1, shuffle=True, num_workers=0)
example ,label= next(iter(interpretation_loader))
#example = test_images[2][0]


image_output = np.transpose(example.squeeze().cpu().detach().numpy(), (1,2,0))
image_output+=1
print(image_output.max(),image_output.min())

attribution = occlusion.attribute(example,
                                       strides = strides,
                                       target=target,
                                       sliding_window_shapes=sliding_window_shapes,
                                       baselines=baselines)

print("Attribution complete")

from captum.attr import visualization as viz
# Convert the compute attribution tensor into an image-like numpy array
attribution_output = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1,2,0))


vis_types = ["heat_map", "original_image"]
vis_signs = ["all", "all"] # "positive", "negative", or "all" to show both
# positive attribution indicates that the presence of the area increases the prediction score
# negative attribution indicates distractor areas whose absence increases the score

"""_ = viz.visualize_image_attr_multiple(attribution_output,
                                      image_output,
                                      vis_types,
                                      vis_signs,
                                      show_colorbar = True)"""

viz.visualize_image_attr_multiple(attribution_output,original_image=image_output,signs=["all", "positive", "negative"],methods=["original_image", "blended_heat_map","blended_heat_map"])