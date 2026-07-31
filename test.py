from PIL import Image as PilImage # not currently used, kept in case need to use imshow to output images
import numpy as np
import skimage as ski
import tifffile as ti # import and export of tif files
import os

path = (".")



# walk through given path, find all tif files and add to file_list
# NOTE: this also includes subfolders
file_list = []
for (dirpath, dirnames, filenames) in os.walk(path):
    for file in filenames:
        if file[len(file) - 3:] == "tif":
            file_list.append(os.path.join(dirpath, file).replace("\\","/"))


# iterate through all files in folder
for file in file_list:
    # input image to np array in the form slice, channel, y, x
    input_image = ti.imread(file)

    #print(input_image.shape)
    n_slices, n_channels, height, width = input_image.shape

    # define variables for segmenting clones
    GFP_channel = 1
    sigma = 10
    open = 2
    dilation_radius = 5
    min_area = 10000

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

    # label image, remove any labelled regions below size threshold then relabel sequentially
    labelled_image = ski.morphology.label(thresholded_image, connectivity=1)
    labelled_image = ski.morphology.remove_small_objects(labelled_image, max_size = min_area)
    labelled_image,_,_ = ski.segmentation.relabel_sequential(labelled_image)

    # define np array of images of all output regions in the format: labelled region, channel, y, x
    output_images = np.zeros((labelled_image.max(),n_channels,height,width))

    # for isolating regions, transpose output images and maxprojected images from (label), channel, y, x to (label), y, x channel
    output_images = output_images.transpose((0,2,3,1))
    maxproject = maxproject.transpose(1,2,0)

    # go through each cell of the labelled image - if the value is not 0 (ie, there is a label there) add all channel values from max projected input
    # to output image for relevant labelled region
    for y, row in enumerate(labelled_image):
        for x, column in enumerate(row):
            if column != 0:
                output_images[column - 1][y][x] = maxproject[y][x]

    # undo transposition for output images and maxproject, leaving them in shape (region), channel, y, x
    output_images = output_images.transpose(0,3,1,2)
    maxproject = maxproject.transpose(2,0,1)

    #output images to multi dimensional tifs
    for n, image in enumerate(output_images):
        output_file_name = file[:len(file) - 4] + " " + str(n) + ".tif"
        ti.imwrite(output_file_name, image, photometric='minisblack')

for channel in output_images[1]:
    PilImage.fromarray(channel).show()
