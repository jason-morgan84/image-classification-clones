import numpy as np
import math

def triangle_unnormalised(image):
    # Implementation of triangle algorithm for thresholding.
    # ImageJ and Skimage give different values for triangle thresholding the same image.
    # This appears to be due to Skimage normalising the histogram bin widths and ImageJ not doing so.
    # In order to a have an implementation of triangle algorithm that matches ImageJ, this function does not normalise bind widths.

    """Zack GW, Rogers WE, Latt SA. Automatic measurement of sister chromatid exchange frequency. J Histochem Cytochem. 1977 Jul;25(7):741-53. doi: 10.1177/25.7.70454. PMID: 70454."""

    # NOTE - THIS ONLY WORKS WHEN PEAK IS ON THE LEFT OF THE HISTOGRAM

    # get histogram of image
    
    histogram, _ = np.histogram(image, 256, (0, 256))
    #print(np.histogram(image, 255, (0, 255)))
    peak_height = histogram.max()
    peak_index = histogram.argmax()

    largest_bin_with_value = 0

    for n, bin in enumerate(histogram):
        if bin != 0:
            largest_bin_with_value = n

    min_height = histogram[largest_bin_with_value]



    max_vertical_distance = 0
    threshold_bin = 0



    print("peak_height", peak_height)
    print("peak_index", peak_index)
    print("largest_bin_with_value", largest_bin_with_value)
    print("difference per bin", peak_height/(largest_bin_with_value - peak_index))
    difference_per_bin = peak_height/(largest_bin_with_value - peak_index)
    # y = mx + c
    m = (min_height - peak_height)/(largest_bin_with_value - peak_index)
    m_n = -1/m

    c = peak_height - m * peak_index
    print("y = ",m,"x + ",c)

    for i in range(peak_index, largest_bin_with_value, 1):
        # This calculates the vertical distance between the diagonal between the peak histogram height (normalised to 1) and the histogram width (normalised to 1)
        # Techincally, the triangle algorithm needs the perpendicular distance between the line and the top of the histogram bar being analysed, but this is proportional
        # to the vertical distance (sin45 * verticaldistance)

        c_n = histogram[i] - i * m_n

        # calcualte intersection between y = mx + c and y = m_nx + c_n
        x = (c_n - c) / (m - m_n)
        y = m * x + c

        distance = math.sqrt((x - i)**2 + (y - histogram[i])**2)


        
        #vertical_distance = (peak_height - (i - peak_index) * difference_per_bin) - histogram[i]

        #vertical_distance = 1 - ((i - peak_index) - histogram[i] / peak_height)
        print(i, distance, histogram[i])

        if distance > max_vertical_distance:
            max_vertical_distance = distance
            threshold_bin = i

    print(threshold_bin)


