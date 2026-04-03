from captum.attr import Occlusion
from torch.utils.data import DataLoader
import numpy as np

def model_occlusion(model,test_images):
    model.eval()

    occlusion = Occlusion(model)
    strides = (3, 9, 9)  # smaller = more fine-grained attribution but slower
    target = 0,  # AR index
    sliding_window_shapes = (3, 15, 15)  # choose size enough to change object appearance
    baselines = 0  # values to occlude the image with 0 corresponds to gray

    interpretation_loader = DataLoader(test_images, batch_size=1, shuffle=True, num_workers=0)
    example, label = next(iter(interpretation_loader))
    # example = test_images[2][0]

    image_output = np.transpose(example.squeeze().cpu().detach().numpy(), (1, 2, 0))
    image_output += 1

    attribution = occlusion.attribute(example,
                                      strides=strides,
                                      target=target,
                                      sliding_window_shapes=sliding_window_shapes,
                                      baselines=baselines)

    print("Attribution complete")

    from captum.attr import visualization as viz
    # Convert the compute attribution tensor into an image-like numpy array
    attribution_output = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))

    vis_types = ["heat_map", "original_image"]
    vis_signs = ["all", "all"]  # "positive", "negative", or "all" to show both
    # positive attribution indicates that the presence of the area increases the prediction score
    # negative attribution indicates distractor areas whose absence increases the score

    """_ = viz.visualize_image_attr_multiple(attribution_output,
                                          image_output,
                                          vis_types,
                                          vis_signs,
                                          show_colorbar = True)"""
    viz.visualize_image_attr_multiple(attribution_output, original_image=image_output,
                                      signs=["all", "positive", "negative"],
                                      methods=["original_image", "blended_heat_map", "blended_heat_map"])
