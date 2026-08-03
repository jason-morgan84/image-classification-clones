from PIL import Image as PilImage # not currently used, kept in case need to use imshow to output images
import numpy as np
import skimage as ski
import tifffile as ti # import and export of tif files
import os

path = (".//working")
path = (".//")

#define output options
#TODO: update output size to take into account desired channels etc
channels = (2, 4)
output_size = (4, 512, 512)
output_cropped = True
pad_cropped = True

# define variables for segmenting clones
GFP_channel = 1
sigma = 8
dilation_radius = 5
min_area = 10000


# walk through given path, find all tif files and add to file_list
file_list = []
for file in os.listdir(path):
    if file[len(file) - 3:] == "tif":
        file_list.append(file)

# iterate through all files in folder
for file in file_list:
    print(file)
    # input image to np array in the form slice, channel, y, x
    input_image = ti.imread(path+"//"+file)

    n_slices, n_channels, height, width = input_image.shape

    # max project image and get max projected GFP channel for segmentation
    maxproject = np.max(input_image, axis=0)
    maxproject_for_threshold = maxproject[GFP_channel]

    # apply gaussian filter then get threshold value and threshold
    maxproject_gauss = ski.filters.gaussian(maxproject_for_threshold, sigma) * 255 
    threshold = ski.filters.threshold_triangle(maxproject_gauss)
    thresholded_image = maxproject_gauss > threshold

    # dilate thresholded image to include surrounding regions and join up clones with small gaps that are likely to be single clones
    # then remove holes within thresholded regions
    thresholded_image = ski.morphology.dilation(thresholded_image,footprint=[(np.ones((dilation_radius, 1)), 1), (np.ones((1, dilation_radius)), 1)])
    thresholded_image = ski.morphology.remove_small_holes(thresholded_image)

    #PilImage.fromarray(thresholded_image).show()

    # label image, remove any labelled regions below size threshold then relabel sequentially
    labelled_image = ski.morphology.label(thresholded_image, connectivity=1)
    labelled_image = ski.morphology.remove_small_objects(labelled_image, max_size = min_area)
    labelled_image,_,_ = ski.segmentation.relabel_sequential(labelled_image)

    label_properties = ski.measure.regionprops(labelled_image)

    # define np array of images of all output regions in the format: labelled region, channel, y, x
    isolated_segments = np.zeros((labelled_image.max(),n_channels,height,width))

    # for isolating regions, transpose output images and maxprojected images from (label), channel, y, x to (label), y, x channel
    isolated_segments = isolated_segments.transpose((0,2,3,1))
    maxproject = maxproject.transpose(1,2,0)

    # go through each cell of the labelled image - if the value is not 0 (ie, there is a label there) add all channel values from max projected input
    # to output image for relevant labelled region
    for y, row in enumerate(labelled_image):
        for x, column in enumerate(row):
            if column != 0:
                isolated_segments[column - 1][y][x] = maxproject[y][x]

    #if output is cropped to segments
    if output_cropped == True:
        output_images=[]

        #gets properties of all labels
        label_properties = ski.measure.regionprops(labelled_image)
        for n, label in enumerate(label_properties):
            # for each label, gets bounding box and crops image to that region
            min_row, min_col, max_row, max_col = label.bbox
            new_image = np.array(isolated_segments[n][min_row:max_row, min_col:max_col])

            # if centre cropped image is true, gets how much the images need padding to match original dimensions, transposes back to CYX
            # then goes through each channel and pads the image
            if pad_cropped == True:
                padding_y = height - (max_row - min_row)
                padding_x = width - (max_col - min_col)
                new_image = new_image.transpose(2,0,1)
                padded_image=[]
                for channel in new_image:
                    new_channel = np.pad(channel,((padding_y//2, padding_y//2 + padding_y%2), (padding_x//2, padding_x//2 + padding_x%2)))
                    padded_image.append(new_channel)

                output_images.append(np.array(padded_image))
            else:
                output_images.append(new_image.transpose(2,0,1))


    else:
        output_images = isolated_segments.transpose(0,3,1,2)


    # undo transposition for output images and maxproject, leaving them in shape (region), channel, y, x
    isolated_segments = isolated_segments.transpose(0,3,1,2)
    maxproject = maxproject.transpose(2,0,1)


    if output_size != (height, width):
        print("resize")




    #output images to multi dimensional tifs
    for n, image in enumerate(output_images):
        output_file_name = file[:len(file) - 4] + " " + str(n) + ".tif"
        print(output_file_name)
        ti.imwrite(os.path.join("./output",output_file_name), ski.transform.resize(image,output_size), photometric='minisblack', metadata={"axes": "CYX"})

#for channel in output_images[1]:
    #PilImage.fromarray(channel).show()
