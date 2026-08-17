from captum.attr import Occlusion
import numpy as np
import os

import torch
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torch.autograd import Variable

from captum.attr import visualization as viz
import classification_class


def model_occlusion(model,test_image,label, n_channels):
    model.eval()
    occlusion = Occlusion(model)
    strides = (n_channels, 9, 9)  # smaller = more fine-grained attribution but slower
    target = label,  # AR index
    sliding_window_shapes = (n_channels, 15, 15)  # choose size enough to change object appearance
    baselines = 0  # values to occlude the image with 0 corresponds to gray

    image_output = test_image.squeeze().cpu().detach().numpy()
    _, height, width = image_output.shape
    #image_output = image_output * 0.05 + 0.04
    image_output = np.transpose(np.append(image_output, np.zeros((1,height,width)), axis=0),(1,2,0)).astype(np.uint8)
    #image_output = np.transpose(image_output,2,0,1)




    attribution = occlusion.attribute(test_image,
                                      strides = strides,
                                      target = target,
                                      sliding_window_shapes = sliding_window_shapes,
                                      baselines = baselines)

    print("Attribution complete")


    # Convert the compute attribution tensor into an image-like numpy array
    attribution_output = attribution.squeeze().cpu().detach().numpy()
    attribution_output = np.transpose(np.append(attribution_output, np.zeros((1,height,width)), axis=0),(1,2,0))

    # positive attribution indicates that the presence of the area increases the prediction score
    # negative attribution indicates distractor areas whose absence increases the score

    viz.visualize_image_attr_multiple(attribution_output,
                                      original_image = image_output,
                                      signs=["all", "positive", "negative"],
                                      methods=["original_image", "blended_heat_map", "blended_heat_map"])

def get_image_predictions(model, device, loader, groups):

    model.to(device)
    model.eval()

    image_predictions=[]

    with torch.no_grad():
        for data in loader:
            images = Variable(data['image'].to(device))
            labels = Variable(data['image_class'].to(device))
            names = data['file_name']

            # run the model on the test set to predict labels
            outputs = model(images.to(device))

            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            for n, prediction in enumerate(predicted):
                image_predictions.append([names[n],prediction.item(),labels[n].item()])

    return image_predictions

def calculate_class_accuracy(predictions, groups):
    two_way_table = np.zeros((len(groups), len(groups)))
    group_accuracy = np.zeros(len(groups))
    correct = 0

    for item in predictions:
        genotype = item[2]
        predicted = item[1]
        two_way_table[genotype][predicted] += 1
        if predicted == genotype:
            correct += 1

    overall_accuracy = float(round(correct / two_way_table.sum() * 100, 2))
    print("Overall accuracy:", str(overall_accuracy) + "%")

    for n, genotype in enumerate(two_way_table):
        group_accuracy[n] = genotype[n] / sum(genotype)
        print(groups[n] + ": " + str(round(group_accuracy[n] * 100,2)) + "%")


    return two_way_table, overall_accuracy, group_accuracy

def test_item(model,device,image):
    model.eval()
    model.to(device)
    output = model(image.to(device))
    _,predicted = torch.max(output.data, 1)
    return predicted

# testing definitions
test_batch_size = 5

# model definitions
model_location = "/mnt/74C88A6CC88A2D04/Lab/Classification/models/modelD20260817T140253"

# get settings used to train model from saved model
settings = dict()
with open(os.path.join(model_location, "settings.txt"), "r") as file:

    for line in file.readlines():
        key, value = line.rstrip().split("\t")
        #TODO: look into replacing eval with JSON https://docs.python.org/2/library/json.html
        settings[key] = eval(value)


# setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

# load model
model = classification_class.ResNet50(num_classes = len(settings['classes']), channels = settings['n_channels'])
model.load_state_dict(torch.load(os.path.join(model_location, "model.pt")))

model.to(device)
print('Loaded model')

# set up testing images and dataloader
testing_images = classification_class.ImagesDataset(annotations_file = 'testing.csv',
                                                 img_dir = os.path.join(settings['image_location'], "testing/"),
                                                 transform = Compose(settings["transformations"]))

test_loader = DataLoader(dataset = testing_images,
                         batch_size = test_batch_size,
                         shuffle = True,
                         num_workers = 0)

# Get predictions for each image in test dataset, then calculate 2 way table, overall accuracy and class accuracy

predictions = get_image_predictions(model, device, test_loader, settings['classes'])
two_way_table, accuracy, group_accuracy = calculate_class_accuracy(predictions, settings['classes'])

# output accuracys and predictions
with open(os.path.join(model_location, "test_results.txt"), "w") as file:
    write_line = '\t' + "\t".join(group for group in settings['classes']) +'\n'
    file.write(write_line)
    for n, genotype in enumerate(two_way_table):
        write_line = settings['classes'][n] + "\t"+"\t".join(str(item) for item in genotype) +'\n'
        file.write(write_line)
    file.write("\t".join(["overall_accuracy",str(accuracy)])+'\n')
    for n, genotype in enumerate(group_accuracy):
        write_line = settings['classes'][n] + "\t" + str(round(genotype*100,2)) +"\n"
        file.write(write_line)

with open(os.path.join(model_location, "predictions.csv"), "w") as file:
    file.write("\t".join(("name","prediction","genotype"))+'\n')
    for line in predictions:
        file.write(line[0] + "\t" + str(settings['classes'][line[1]]) + "\t" + str(settings['classes'][line[1]]) + '\n')


# Test batch of images
batch = next(iter(test_loader))
prediction = test_item(model, device, batch["image"])

for j in range(test_batch_size):
    print('File: {}, Real Label: {}, Predicted Label: {}'.format(batch["file_name"][j],batch["genotype"][j],settings['classes'][prediction[j]]))

# dispaly interpreted image
print("Interpretation")
test_loader = DataLoader(dataset = testing_images,
                         batch_size = 1,
                         shuffle = True,
                         num_workers = 0)

batch = next(iter(test_loader))
prediction = test_item(model, device, batch["image"])
print('File: {}, Real Label: {}, Predicted Label: {}'.format(batch["file_name"][0],batch["genotype"][0],settings['classes'][prediction.item()]))

model_occlusion(model, batch["image"].to(device), batch["image_class"].to(device).item(), settings['n_channels'])