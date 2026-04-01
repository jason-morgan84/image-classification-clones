import time
from torch.optim import Adam
from torch.autograd import Variable
import torch
import torch.nn as nn

import model_test

def train(model, device, train_loader, test_loader, num_epochs):
    model.train()
    model.to(device)

    n_batches = image_classification.n_batches
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()

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
            if i % int((n_batches - 1 ) /2) == int((n_batches - 1 ) /2 - 1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / n_batches))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = model_test.test_accuracy(model, device, test_loader)
        print('For epoch', epoch + 1 ,'the test accuracy over the whole test set is %d %%' % accuracy ,"("
              ,round((time.time() - start_time) ,3) ,"seconds)")

        # we want to save the model if the accuracy is the best
        """if accuracy > best_accuracy:
            save_model()
            best_accuracy = accuracy"""

    torch.save(model.state_dict(), "./myFirstModel.pth")
