import time
from torch.optim import Adam
from torch.autograd import Variable
import torch
import torch.nn as nn
from torchvision.transforms import Compose,Normalize,RandomRotation,Resize

import model_test



def train(model, device, train_loader, test_loader, num_epochs,n_batches):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for data in train_loader:
            images = Variable(data['image'].to(device))
            labels = Variable(data['image_class'].to(device))

            #normalize by batch
            sum_mean = 0
            sum_sd = 0
            for image in images:
                mean_sd = torch.std_mean(image)
                sum_mean += mean_sd[0].item()
                sum_sd += mean_sd[1].item()
            mean = sum_mean/len(images)
            sd = sum_sd/len(images)
            transform_norm = Compose([Normalize(mean, sd)])

            # get normalized image
            #for image in images:
               # image = transform_norm(image)


            # zero the parameter gradients
            optimizer.zero_grad()

            # predict classes using images from the training set
            outputs = model(transform_norm(images))
            # compute the loss based on model output and real labels
            loss = criterion(outputs, labels)

            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy, val_loss = model_test.test_accuracy(model, device, test_loader)

        print('Epoch {}: Training loss - {}, validation loss - {}, accuracy - {}, time taken - {} seconds' .format(epoch+1,round(running_loss/len(train_loader),3),round(val_loss,3),round(accuracy,0),round(time.time()-start_time),0))

        # we want to save the model if the accuracy is the best
        #if accuracy > best_accuracy:
            #torch.save(model.state_dict(), "./ResNet10 50.pth")
            #best_accuracy = accuracy



