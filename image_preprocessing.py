from PIL import Image as PilImage # not currently used, kept in case need to use imshow to output images
import numpy as np
import skimage as ski
import tifffile as ti # import and export of tif files
import os
import czifile



class file_data():
    def __init__(self, genotype, date, name, location, **kwargs):
        self.genotype = genotype
        self.date = date
        self.name = name
        self.location = location
        self.ID = kwargs.get('ID',None)


    def __str__(self):
        return ", ".join((self.genotype,self.date,self.name,self.location))

def select_channels(image, output_channels, channel_of_interest):


    channels, _, _ = image.shape
    channels_for_deletion = []
    channel_of_interest_reduction = 0
    for i in range(channels):
        if i not in output_channels:
            channels_for_deletion.append(i)
            if i < channel_of_interest: channel_of_interest_reduction += 1
    channel_of_interest -= channel_of_interest_reduction
    
    return np.delete(image,channels_for_deletion, axis = 0), channel_of_interest, len(output_channels)

def segment (image, channel, sigma, dilation_radius, min_area):

    image_channel = image[channel]

    # apply gaussian filter then get threshold value and threshold
    image_gauss = ski.filters.gaussian(image_channel, sigma) * 255 
    #triangle in imagej consistently gives 2-3 higher threshold than through other algorithms
    threshold = ski.filters.threshold_otsu(image_gauss) + 2 
    image_thresholded = image_gauss > threshold

    # dilate thresholded image to include surrounding regions and join up clones with small gaps that are likely to be single clones
    # then remove holes within thresholded regions
    image_thresholded = ski.morphology.dilation(image_thresholded,footprint=[(np.ones((dilation_radius, 1)), 1), (np.ones((1, dilation_radius)), 1)])
    image_thresholded = ski.morphology.remove_small_holes(image_thresholded)

    #PilImage.fromarray(image_thresholded).show()

    # label image, remove any labelled regions below size threshold then relabel sequentially
    image_labelled = ski.morphology.label(image_thresholded, connectivity = 1)
    image_labelled = ski.morphology.remove_small_objects(image_labelled, max_size = min_area)
    image_labelled,_,_ = ski.segmentation.relabel_sequential(image_labelled)

    return image_labelled

def isolate_segments (image, image_labelled, height, width, n_channels): # outputs in format syxc

    n_labels = image_labelled.max()   
    isolated_segments = [np.zeros((height, width, n_channels)) for _ in range(n_labels)]

    # define np array of images of all output regions in the format: labelled region, channel, y, x
    #isolated_segments = np.zeros((image_labelled.max(), height, width, n_channels))

    # for isolating regions, transpose input images from cyx to yxc
    image = image.transpose(1,2,0)

    # go through each cell of the labelled image - if the value is not 0 (ie, there is a label there) add all channel values from max projected input
    # to output image for relevant labelled region
    for y, row in enumerate(image_labelled):
        for x, column in enumerate(row):
            if column != 0:
                isolated_segments[column - 1][y][x] = image[y][x]

    return isolated_segments

def crop_segments (image_segments, # 4D np array of original image divided into segmented regions in the format syxc
                   label_properties): # properties of segmented regions

    output_images=[]

    #gets properties of all labels
    
    for n, label in enumerate(label_properties):
        # for each label, gets bounding box and crops image to that region
        min_row, min_col, max_row, max_col = label.bbox
        new_image = np.array(image_segments[n][min_row:max_row, min_col:max_col])
        output_images.append(np.array(new_image))

    return output_images

def pad_cropped_segments (image_segments, 
                          height, 
                          width, 
                          label_properties):
        # if centre cropped image is true, gets how much the images need padding to match original dimensions, transposes back to CYX
        # then goes through each channel and pads the image
        output_images = []
        for n, label in enumerate(label_properties):
            min_row, min_col, max_row, max_col = label.bbox

            padding_y = height - (max_row - min_row)
            padding_x = width - (max_col - min_col)
            padded_image=[]
            for channel in image_segments[n].transpose(2, 0, 1):
                new_channel = np.pad(channel, 
                                     ((padding_y // 2, padding_y // 2 + padding_y % 2), 
                                      (padding_x // 2, padding_x // 2 + padding_x % 2)))
                padded_image.append(new_channel)

            output_images.append(padded_image)
        return np.array(output_images).transpose(0, 2, 3, 1)

def resize (image, size, channels):
    output_images = []
    for item in image:
        output_images.append(ski.transform.resize(item.transpose(2, 0, 1), (channels,) + size))
    return output_images

# file location variables
path_input = "D:\\Lab\\Classification\\All Images\\Input File Lists\\ar_ars_file_list.csv"
path_output_csv = "D:\\Lab\\Classification\\All Images\\Processed 260805"
path_output_images = "D:\\Lab\\Classification\\All Images\\Processed 260805"

# output options
channels = (1, 3)
output_size = (256, 256)

# input variables
GFP_channel = 1

# segmentation variables
sigma = 8
dilation_radius = 10
min_area = 10000

input_file_list = []
#open csv with file information and process into list of file_data class
with open(path_input, "r") as input_file:
    _ = input_file.readline()

    for line in input_file:
        genotype, date, name, location = tuple(line.rstrip().split(","))
        new_file = file_data(genotype, date, name, location)
        input_file_list.append(new_file)

if not os.path.exists(path_output_images):
    os.makedirs(path_output_images)

# iterate through all files in folder

with open(os.path.join(path_output_csv,"output.csv"), "w") as output_file:
    output_file.writelines("genotype,date,name,input_location,clone,output_location\n")

for file_number, file in enumerate(input_file_list):
    print(file.location, "(",file_number,"of",len(input_file_list),")")
    # input image to np array in the form slice, channel, y, x
    ending = file.location[len(file.location)-4:]
    try:
        if ending == ".tif" or ending == "tiff":
            input_image = ti.imread(file.location)
        elif ending ==".czi":
            input_image = czifile.imread(file.location).transpose(1, 0, 2, 3)
        else:
            print("Unsupported file type")
            continue
    except:
        print("File cannot be opened")

    n_slices, n_channels, height, width = input_image.shape

    # max project image and get max projected GFP channel for segmentation {cyx}
    maxproject = np.max(input_image, axis = 0)

    # remove channels that aren't of interest {cyx}
    maxproject_sliced, GFP_channel, n_channels = select_channels(maxproject, channels, GFP_channel)
 
    # label max projected image {yx} and get properties of labels
    labelled_image = segment(maxproject_sliced, GFP_channel, sigma, dilation_radius, min_area) 
    label_properties = ski.measure.regionprops(labelled_image)

    # for each segment, returns image with only content in the the region corresponding to that segment
    # returns a list of regions. each region is a numpy array in the format {yxc}
    isolated_segments = isolate_segments(maxproject_sliced, labelled_image, height, width, n_channels)
    if len(isolated_segments) == 0:
        print("No clones found")
        continue

    # for each isolated segment, crops down to the bounding box of that segment
    cropped_segments = crop_segments(isolated_segments, label_properties)

    # pads cropped segments so isolated region is central
    padded_segments = pad_cropped_segments(cropped_segments, height, width, label_properties)

    # resizes image to desired dimensions
    output_image = resize(padded_segments, output_size, n_channels) 

    #output images to multi dimensional tifs
    with open(os.path.join(path_output_csv,"output.csv"), "a") as output_file:
        for n, image in enumerate(output_image):
            output_file_name = str(file_number) + "c" + str(n) + ".tif"
            print(output_file_name)
            ti.imwrite(os.path.join(path_output_images, output_file_name), image, photometric='minisblack', metadata={"axes": "CYX"})
            output_file.writelines(",".join([file.genotype, file.date, file.name, file.location, str(n), os.path.abspath(os.path.join(path_output_images,output_file_name))+"\n"]))
    
#genotype, date, name, input_location, clone, output_location

        
#for channel in output_images[1]:
    #PilImage.fromarray(channel).show()
