import torch
from sympy import true
from torch.autograd import Variable


def test_item(model,device,image):
    model.eval()
    model.to(device)
    output = model(image.to(device))
    _,predicted = torch.max(output.data, 1)
    return predicted

# Function to test the model with the test dataset and print the accuracy for the test images
def test_accuracy(model, device, test_loader):
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels, _ = data
            # run the model on the test set to predict labels
            outputs = model(images.to(device))
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy


def test_class_accuracy(model, device, test_loader, groups):

    model.to(device)
    model.eval()
    n_groups = len(groups)
    group_total = [0] * n_groups
    group_accuracy = []
    total = [0] * n_groups

    with torch.no_grad():
        for data in test_loader:
            images, labels, _ = data
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
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

        for n,group in enumerate(groups):
            group_accuracy.append((group_total[n]/total[n])*100)
            print(group+" ("+str(n)+") accuracy: "+str(round(group_accuracy[n],2))+"%")
    return group_accuracy

 # Function to test the model with a batch of images and show the labels predictions

"""def test_batch(model, device, test_loader, groups):
    # get batch of images from the test DataLoader
    model.eval()
    images, labels = next(iter(test_loader))
    #batch_size = 5

    # show all images as one image grid
    #imagegrid = torchvision.utils.make_grid(images)
    #print(imagegrid.size())
    #imagegrid = imagegrid.transpose(1,2,0)
    #img = torchvision.transforms.ToPILImage()(imagegrid)
    #img.show()

    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % groups[labels[j]] for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    model.to(device)
    outputs = model(images.to(device))

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs.data, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % groups[predicted[j]]
                                  for j in range(batch_size)))"""