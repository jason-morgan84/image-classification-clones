import torch
import classification_class
from torchvision.transforms import Compose,Normalize
from torch.utils.data import DataLoader
from captum.attr import Occlusion
import numpy as np

import model_test
import model_train



#-1: what does captum output mean?
#0: look into loss, training loss v validation loss
#1: Fix artifacts in outputted images?
#2 look into more confocal specific models
#3 how to choose the cnn architecture?


batch_size = 5
classes = ("AR","ARS")

# Define your execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

print("The model will be running on", device, "device")

transform_norm = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_images = classification_class.ImagesDataset(annotations_file='images.csv', img_dir='samples/',transform=transform_norm)
#n_train_images = len(train_images)
n_batches = len(train_images)/batch_size

test_images = classification_class.ImagesDataset(annotations_file='test_images.csv', img_dir='test/',transform=transform_norm)

train_loader=DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader=DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=0)

model = classification_class.Network()




example,label= next(iter(train_loader))
# Let's build our model
#model_train.train(model, device, train_loader, test_loader, 10)
print('Finished Training')



# Let's load the model we just created and test the accuracy per label
path = "myFirstModel.pth"
model.load_state_dict(torch.load(path))
print('Loaded model')

#test_class_accuracy(classes)

# Test with batch of images
#test_batch()



model.eval()

occlusion = Occlusion(model)
strides = (3,9, 9)               # smaller = more fine-grained attribution but slower
target=0,                       # AR index
sliding_window_shapes=(3,15, 15)  # choose size enough to change object appearance
baselines = 0                     # values to occlude the image with 0 corresponds to gray

interpretation_loader=DataLoader(test_images, batch_size=1, shuffle=True, num_workers=0)
example ,label= next(iter(interpretation_loader))
#example = test_images[2][0]


image_output = np.transpose(example.squeeze().cpu().detach().numpy(), (1,2,0))
image_output+=1
print(image_output.max(),image_output.min())

attribution = occlusion.attribute(example,
                                       strides = strides,
                                       target=target,
                                       sliding_window_shapes=sliding_window_shapes,
                                       baselines=baselines)

print("Attribution complete")

from captum.attr import visualization as viz
# Convert the compute attribution tensor into an image-like numpy array
attribution_output = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1,2,0))


vis_types = ["heat_map", "original_image"]
vis_signs = ["all", "all"] # "positive", "negative", or "all" to show both
# positive attribution indicates that the presence of the area increases the prediction score
# negative attribution indicates distractor areas whose absence increases the score

"""_ = viz.visualize_image_attr_multiple(attribution_output,
                                      image_output,
                                      vis_types,
                                      vis_signs,
                                      show_colorbar = True)"""

viz.visualize_image_attr_multiple(attribution_output,original_image=image_output,signs=["all", "positive", "negative"],methods=["original_image", "blended_heat_map","blended_heat_map"])