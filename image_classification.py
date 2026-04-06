import torch
import classification_class
from torchvision.transforms import Compose,Normalize,RandomRotation
from torch.utils.data import DataLoader

import model_interpret
import model_test
import model_train

#-1: Gradient outputs - find exploding or vanishing gradients
#0: more training data
#1: what does captum output mean?
########: 1.2: Allow testing/interepretation of chosen images
########: 1.3: Test other methods of interpretation
########: 1.4: Output probability for given image
#3: Fix artifacts in outputted images?
#4 look into more confocal specific models
#5 how to choose the cnn architecture?

training_batch_size = 16
test_batch_size = 5
classes = ("AR","ARS")
num_epochs = 20


# Define your execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("The model will be running on", device, "device")

#transform_norm = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_norm = Compose([RandomRotation(180),Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

train_images = classification_class.ImagesDataset(annotations_file='images.csv', img_dir='samples/',transform=transform_norm)
n_batches = len(train_images)/training_batch_size

test_images = classification_class.ImagesDataset(annotations_file='test_images.csv', img_dir='test/',transform=transform_norm)

model = classification_class.VGG16()
print("Model setup")


# Let's build our model
train_loader=DataLoader(train_images, batch_size=training_batch_size, shuffle=True, num_workers=0)
test_loader=DataLoader(test_images, batch_size=test_batch_size, shuffle=True, num_workers=0)
#print(len(train_loader))
print("Images Loaded")
model_train.train(model, device, train_loader, test_loader, num_epochs, n_batches)
print('Finished Training')


# Let's load the model we just created and test the accuracy per label
path = "myFirstModel.pth"
model.load_state_dict(torch.load(path))
print('Loaded model')





# Get accuracy of classes:
test_loader=DataLoader(test_images, batch_size=test_batch_size, shuffle=True, num_workers=0)
model_test.test_class_accuracy(model, device, test_loader, classes)

# Test batch of images
test_loader=DataLoader(test_images, batch_size=test_batch_size, shuffle=True, num_workers=0)
images, labels, file_names = next(iter(test_loader))
prediction = model_test.test_item(model, device, images)
print('Real labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(test_batch_size)))
print('Predicted: ', ' '.join('%5s' % classes[prediction[j]] for j in range(test_batch_size)))


test_loader=DataLoader(test_images, batch_size=1, shuffle=True, num_workers=0)
image, label, file_name = next(iter(test_loader))
prediction = model_test.test_item(model, device, image)
print(file_name[0],"is",classes[label.item()],"and is predicted to be",classes[prediction.item()])
model_interpret.model_occlusion(model, image.to(device))



