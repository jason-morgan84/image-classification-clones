import time

from torch.autograd import Variable
import torch

from torchvision.transforms import Compose,Normalize,RandomRotation,Resize




def test_accuracy(model,
                  loss_function,
                  device,
                  loader):
    model.eval()
    accuracy = 0.0
    total = 0.0
    val_loss = 0
    with torch.no_grad():
        for data in loader:
            #print("Testing batch: ", n)
            images = Variable(data['image'].to(device))
            labels = Variable(data['image_class'].to(device))

            # run the model on the test set to predict labels
            outputs = model(images)
            loss = loss_function(outputs, labels)
            val_loss += loss.item() * labels.size(0)

            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()

    # compute the accuracy over all test images
    val_loss /= len(loader.dataset)
    accuracy = (100 * accuracy / total)
    return accuracy, val_loss

def train(model,
          loss_function,
          optimizer,
          device,
          train_loader,
          validation_loader,
          num_epochs,
          batch_size_train,
          batch_size_validation):
    model.to(device)

    n_batches_train = len(train_loader)
    n_batches_validation = len(validation_loader)

    training_loss_history = []
    validation_loss_history = []

    training_accuracy_history = []
    validation_accuracy_history = []


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for n, data in enumerate(train_loader):
            print("\r","Epoch", epoch + 1, "of", num_epochs,". Batch", n,"of", n_batches_train, end = "")
            images = Variable(data['image'].to(device))
            labels = Variable(data['image_class'].to(device))

            # zero the parameter gradients
            optimizer.zero_grad()

            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels

            loss = loss_function(outputs, labels)

            # backpropagate the loss
            loss.backward()

            # adjust parameters based on the calculated gradients
            optimizer.step()

            running_loss += loss.item() * batch_size_train

        # Compute and print the average accuracy for this epoch when tested over all test images
        validation_accuracy, val_loss = test_accuracy(model = model,
                                                                 device = device,
                                                                 loss_function = loss_function,
                                                                 loader = validation_loader)
        training_accuracy, train_loss = test_accuracy(model = model,
                                                                 loss_function = loss_function,
                                                                 device = device,
                                                                 loader = train_loader)

        print('\r','Epoch {}: Training loss - {}, training accuracy - {}, validation loss - {}, validation accuracy - {}, time taken - {} seconds' .format(epoch + 1,
                                                                                                                   round(train_loss, 3),
                                                                                                                   round(training_accuracy, 3),
                                                                                                                   round(val_loss, 3),
                                                                                                                   round(validation_accuracy, 0),
                                                                                                                   round(time.time() - start_time), 0), end = '\n')
        training_loss_history.append(round(train_loss,3))
        training_accuracy_history.append(round(training_accuracy,3))
        validation_loss_history.append(round(val_loss,3))
        validation_accuracy_history.append(round(validation_accuracy,3))

    training_statitics = {"training_loss": training_loss_history,
                          "training_accuracy": training_accuracy_history,
                          "validation_loss": validation_loss_history,
                          "validation_accuracy": validation_accuracy_history}

    return model, training_statitics


