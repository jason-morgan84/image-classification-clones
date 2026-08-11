import torch
from sympy import true
from torch.autograd import Variable



# Function to test the model with the test dataset and print the accuracy for the test images





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