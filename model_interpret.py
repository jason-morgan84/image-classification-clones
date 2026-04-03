from captum.attr import Occlusion
import numpy as np
from captum.attr import visualization as viz

def model_occlusion(model,test_image):

    model.eval()
    occlusion = Occlusion(model)

    strides = (3, 9, 9)  # smaller = more fine-grained attribution but slower
    target = 0,  # AR index
    sliding_window_shapes = (3, 15, 15)  # choose size enough to change object appearance
    baselines = 0  # values to occlude the image with 0 corresponds to gray

    image_output = np.transpose(test_image.squeeze().cpu().detach().numpy(), (1, 2, 0))
    image_output = image_output * 0.5 + 0.5


    attribution = occlusion.attribute(test_image,
                                      strides=strides,
                                      target=target,
                                      sliding_window_shapes=sliding_window_shapes,
                                      baselines=baselines)

    print("Attribution complete")


    # Convert the compute attribution tensor into an image-like numpy array
    attribution_output = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))

    # positive attribution indicates that the presence of the area increases the prediction score
    # negative attribution indicates distractor areas whose absence increases the score

    viz.visualize_image_attr_multiple(attribution_output, original_image=image_output,
                                      signs=["all", "positive", "negative"],
                                      methods=["original_image", "blended_heat_map", "blended_heat_map"])
