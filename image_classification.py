import torch
import classification_class
from torchvision.transforms import Compose,Normalize,RandomRotation,Resize
from torch.utils.data import DataLoader
import os
from torchvision.io import read_image

import model_interpret
import model_test
import model_train


#1: Model testing and analysis
########: 1.3: Test other methods of interpretation
########: 1.4: Output probability for given image


training_batch_size = 15
test_batch_size = 10
classes = ("AR","ARS")
num_epochs = 15


def train_model():
    train_loader = DataLoader(train_images, batch_size=training_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_images, batch_size=test_batch_size, shuffle=True, num_workers=0)
    n_batches = len(train_images) / training_batch_size

    model_train.train(model, device, train_loader, test_loader, num_epochs, n_batches)
    print('Model Trained')

def save_model(file_name):
    torch.save(model.state_dict(),file_name)


# Define your execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform_norm = Compose([RandomRotation(180),Resize((224,224)),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load training and testing datasets
train_images = classification_class.ImagesDataset(annotations_file='images.csv', img_dir='samples/',transform=transform_norm)
test_images = classification_class.ImagesDataset(annotations_file='test_images.csv', img_dir='test/',transform=transform_norm)

#model = classification_class.ResNet(classification_class.ResidualBlock, [3, 4, 6, 3])
model = classification_class.ResNet50(2,3)
#model = classification_class.VGG16()
#model = classification_class.BasicNetwork()
print("Model setup")


# Train and save model
#train_model()
#save_model("./Model.pth")






# Let's load the model we just created and test the accuracy per label
path = "Model.pth"
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


# Model interpretation
test_loader=DataLoader(test_images, batch_size=1, shuffle=True, num_workers=0)
image, label, file_name = next(iter(test_loader))
#path = "samples/"
#file_name = "51.png"
#label = 0

#full_path = os.path.join(path, file_name)

#image = read_image(full_path).float()/255
#image = transform_norm(image).unsqueeze(0)

prediction = model_test.test_item(model, device, image)
print(file_name,"is",classes[label],"and is predicted to be",classes[prediction.item()])
model_interpret.model_occlusion(model, image.to(device), label.to(device).item())



