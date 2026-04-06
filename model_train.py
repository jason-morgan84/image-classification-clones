import time
from torch.optim import Adam
from torch.autograd import Variable
import torch
import torch.nn as nn

import model_test

def train(model, device, train_loader, test_loader, num_epochs,n_batches):
    model.to(device)

    learning_rate = 0.0005 #was 0.005
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    best_accuracy = 0.0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for i, (images, labels, files) in enumerate(train_loader, 0):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy, val_loss = model_test.test_accuracy(model, device, test_loader)

        print('Epoch {}: Training loss - {}, validation loss - {}, accuracy - {}, time taken - {} seconds' .format(epoch+1,round(running_loss/len(train_loader),3),round(val_loss,3),round(accuracy,0),round(time.time()-start_time),0))

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), "./myFirstModel.pth")
            best_accuracy = accuracy


