import datetime
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize

import classification_class
import model_train

#1: Model testing and analysis
########: 1.4: Output probability for given image
#TODO:  Check the correct image data is being used with regard to channels etc
#   What can I change to improve model? Look up optimisation of CNN learning
#       Optimiser
#       Epochs
#       Images
#       Loss function
#       Learning rate

# training settings

training = {
    "image_location": "/mnt/74C88A6CC88A2D04/Lab/Classification/All images/Processed 260814/",
    "classes": ("AR", "ARS"),
    "n_channels": 2,
    "training_batch_size": 15,
    "validation_batch_size" : 10,
    "num_epochs": 30,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "transformations": [],
    "model": "resnet50"
}

# model save location
model_save_name = "model"
model_save_location = "/mnt/74C88A6CC88A2D04/Lab/Classification/models"


# gets mean and std of training images for later normalization
sum_mean = [0.0 for i in range(training["n_channels"])]
sum_std = [0.0 for j in range(training["n_channels"])]
total = 0

with open(os.path.join(training["image_location"], "training/training.csv"),"r") as file:
    file.readline()
    for line in file.readlines():
        _,_,_,_,image_mean,image_std,_,_ = line.rstrip().split(",")
        for n, channel_mean in enumerate(image_mean.split(" ")):
            sum_mean[n] += float(channel_mean)

        for n, channel_std in enumerate(image_std.split(" ")):
            sum_std[n] += float(channel_std)

        total += 1

training_mean = [(sum_mean[i] / total) /255 for i in range(len(sum_mean))]
training_std = [(sum_std[i] / total) /255 for i in range(len(sum_std))]

def save_model(file_name):
    torch.save(model.state_dict(),file_name)

# Define execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device:",device)

# Define image transformations
training["transformations"] = [CenterCrop((224,224)), Normalize(training_mean,training_std)]

# Load training and testing datasets
training_images = classification_class.ImagesDataset(annotations_file='training.csv',
                                                  img_dir = os.path.join(training["image_location"], "training/"),
                                                  transform = Compose(training["transformations"]))

validation_images = classification_class.ImagesDataset(annotations_file='validation.csv',
                                                 img_dir= os.path.join(training["image_location"], "validation/"),
                                                 transform = Compose(training["transformations"]))

train_loader = DataLoader(dataset = training_images,
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

# Save model, model code, key settings and training results
model_save_name += datetime.datetime.now().strftime("D%Y%m%dT%H%M%S")
model_full_path = os.path.join(model_save_location, model_save_name)
os.mkdir(model_full_path)

# save model
torch.save(model.state_dict(), os.path.join(model_full_path,"model.pt"))

# add optimizer and loss_function to training settings dictionary and save settings to file
training.update([("loss_function", loss_function["name"]),("optimizer",optimizer["name"])])
output_settings = ""

for key, value in training.items():
    if type(value) == str:
        output_settings += str(key) + "\t" + "'" + str(value) + "'" + "\n"
    else:
        output_settings += str(key) + "\t" + str(value) + "\n"

with open(os.path.join(model_full_path,"settings.txt"), 'w') as file:
    file.write(output_settings)

# output training results
output_results = "epoch,training_loss,training_accuracy,validation_loss,validation_accuracy\n"
for n in range(training["num_epochs"]):
    output_results += ",".join([str(n),
                               str(training_statistics["training_loss"][n]),
                               str(training_statistics["training_accuracy"][n]),
                               str(training_statistics["validation_loss"][n]),
                               str(training_statistics["validation_accuracy"][n])]) + '\n'

with open(os.path.join(model_full_path,"results.txt"),'w') as file:
    file.write(output_results)

print("Model saved")
















