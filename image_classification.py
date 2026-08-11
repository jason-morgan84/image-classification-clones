import datetime
import os

# from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomRotation, Resize

# from matplotlib.rcsetup import validate_int_or_None
import classification_class
import model_train

#1: Model testing and analysis
########: 1.4: Output probability for given image
#TODO:  Check the correct image data is being used with regard to channels etc
#   What can I change to improve model? Look up optimisation of CNN learning
#       Optimiser
#       Transformations
#           Why are transformations split up?
#       Normalisation
#       Epochs
#       Images
#       Loss function
#       Learning rate

# input definitions

classes = ("AR", "ARS")
n_channels = 2
image_location = "/mnt/74C88A6CC88A2D04/Lab/Classification/All images/Processed 260805/"

# training definitions

training_batch_size = 15
validation_batch_size = 10
num_epochs = 30
lr = 0.001
weight_decay = 0.0001

# output definitions
model_save_name = "model"
model_save_location = "/mnt/74C88A6CC88A2D04/Lab/Classification/models"



def save_model(file_name):
    torch.save(model.state_dict(),file_name)

# Define execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Define image transformations
#transform_norm = Compose([RandomRotation(180),Resize((224,224)),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#transform_norm = Compose([RandomRotation(180),Resize((224,224)),Normalize((0.045, 0.037, 0.5), (0.068, 0.053, 0.5))])
transform_norm = Compose([RandomRotation(180),Resize((224,224))])

# Load training and testing datasets
training_images = classification_class.ImagesDataset(annotations_file='training.csv',
                                                  img_dir = os.path.join(image_location, "training/"),
                                                  transform = transform_norm)

validation_images = classification_class.ImagesDataset(annotations_file='validation.csv',
                                                 img_dir= os.path.join(image_location, "validation/"),
                                                 transform = transform_norm)

train_loader = DataLoader(dataset = training_images,
                          batch_size = training_batch_size,
                          shuffle = True,
                          num_workers = 0)

validation_loader = DataLoader(dataset = validation_images,
                               batch_size = validation_batch_size,
                               shuffle = True,
                               num_workers = 0)




# Set up model
model = classification_class.ResNet50(num_classes = 2, channels = n_channels)
#model = classification_class.ResNet(classification_class.ResidualBlock, [3, 4, 6, 3])
#model = classification_class.VGG16()
#model = classification_class.BasicNetwork()

loss_function = {"name": "CrossEntropyLoss",
                 "function": nn.CrossEntropyLoss()}

optimizer = {"name": "Adam",
             "function": Adam(model.parameters(), lr = lr, weight_decay = weight_decay)}
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

print("Model setup")


print("Training model")
model, training_statistics = model_train.train(model = model,
                          device = device,
                          loss_function = loss_function["function"],
                          optimizer = optimizer["function"],
                          train_loader = train_loader,
                          validation_loader = validation_loader,
                          num_epochs = num_epochs,
                          batch_size_train = training_batch_size,
                          batch_size_validation = validation_batch_size)
print('Model trained')



fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(range(0, num_epochs), training_statistics["training_loss"], label = "Training Loss")
ax1.plot(range(0, num_epochs), training_statistics["validation_loss"], label = "Validation Loss")
ax2.plot(range(0, num_epochs), training_statistics["training_accuracy"], label = "Training Accuracy")
ax2.plot(range(0, num_epochs), training_statistics["validation_accuracy"], label = "Validation Accuracy")

ax1.legend()
ax2.legend()
plt.show()

# Save model and output key data to history file

model_save_name += datetime.datetime.now().strftime("%Y%m%d %H%M%S")
torch.save(model.state_dict(),os.path.join(model_save_location, model_save_name))
print("Model saved")
with open(os.path.join(model_save_location,"history.csv"), "a") as output_file:
    output_file.write(",".join([model_save_name,
                               str(num_epochs),
                               str(training_batch_size),
                               str(loss_function["name"]),
                               str(optimizer["name"]),
                               str(lr),
                               str(weight_decay),
                               image_location]))















# Model interpretation
"""test_loader=DataLoader(test_images, batch_size=1, shuffle=True, num_workers=0)
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
"""


