import datetime
import os
import zipfile
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
#           Currently no normalisation
#       Epochs
#       Images
#       Loss function
#       Learning rate

# training settings

training = {
    "image_location": "/mnt/74C88A6CC88A2D04/Lab/Classification/All images/Processed 260805/",
    "classes": ("AR", "ARS"),
    "n_channels": 2,
    "training_loss": [],
    "validation_loss": [],
    "training_batch_size": 15,
    "validation_batch_size" : 10,
    "num_epochs": 1,
    "lr": 0.001,
    "weight_decay": 0.0001
}

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
                                                  img_dir = os.path.join(training["image_location"], "training/"),
                                                  transform = transform_norm)

validation_images = classification_class.ImagesDataset(annotations_file='validation.csv',
                                                 img_dir= os.path.join(training["image_location"], "validation/"),
                                                 transform = transform_norm)

#TODO update back to training_images
train_loader = DataLoader(dataset = validation_images,
                          batch_size = training["training_batch_size"],
                          shuffle = True,
                          num_workers = 0)

validation_loader = DataLoader(dataset = validation_images,
                               batch_size = training["validation_batch_size"],
                               shuffle = True,
                               num_workers = 0)




# Set up model
model = classification_class.ResNet50(num_classes = 2, channels = training["n_channels"])
#model = classification_class.ResNet(classification_class.ResidualBlock, [3, 4, 6, 3])
#model = classification_class.VGG16()
#model = classification_class.BasicNetwork()

loss_function = {"name": "CrossEntropyLoss",
                 "function": nn.CrossEntropyLoss()}

optimizer = {"name": "Adam",
             "function": Adam(model.parameters(), lr = training["lr"], weight_decay = training["weight_decay"])}
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

print("Model setup")


print("Training model")
model, training_statistics = model_train.train(model = model,
                          device = device,
                          loss_function = loss_function["function"],
                          optimizer = optimizer["function"],
                          train_loader = train_loader,
                          validation_loader = validation_loader,
                          num_epochs = training["num_epochs"],
                          batch_size_train = training["training_batch_size"],
                          batch_size_validation = training["validation_batch_size"])
print('Model trained')



fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(range(0, training["num_epochs"]), training_statistics["training_loss"], label = "Training Loss")
ax1.plot(range(0, training["num_epochs"]), training_statistics["validation_loss"], label = "Validation Loss")
ax2.plot(range(0, training["num_epochs"]), training_statistics["training_accuracy"], label = "Training Accuracy")
ax2.plot(range(0, training["num_epochs"]), training_statistics["validation_accuracy"], label = "Validation Accuracy")

ax1.legend()
ax2.legend()
plt.show()

# Save model and output key data to history file

model_save_name += datetime.datetime.now().strftime("D%Y%m%dT%H%M%S")
model_full_path = os.path.join(model_save_location, model_save_name+".pth")

torch.save(model.state_dict(),model_full_path)
print("Model saved")


# add optimizer and loss_function to training settings dictionary and output to model zip
training.update([("loss_function", loss_function["name"]),("optimizer",optimizer["name"])])
output = ""
for key, value in training.items():
    output += str(key) + "," + str(value) + "\n"
with zipfile.ZipFile(model_full_path,"a") as myzip:
    myzip.writestr("settings.txt",output)


















