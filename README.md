<h1>ResNet image classification of isolated clones by genotype</h1>
<h2>Background</h2>
<h2>Aims</h2>
<h2>Modules and structure</h2>  
<h2>Image pre-processing</h2>
<h2>Input data</h2>
<p>The model receives a folder containing all images required for training, testing and validation in folders named "training", "testing" and "validation".</p>
<p>Images should all be saved as single Z-Stack multi-channel images. The number of channels should be defined in the classification.py, as training["n_channels"]</p>
<p>Each folder must contain a .csv file containing data on the included images. These CSVs have the following columns:</p>
<ul><li>genotype - the genotype the image comes from</li>
  <li>date - the date the image was taken</li>
  <li>name - the file name the clone was extracted from</li>
  <li>clone - the number of the clone from that image</li>
  <li>image_mean - the mean of each channel of that image, separated by a space</li>
  <li>image_std - the standard deviation of each channel of that image, separated by a space</li>
  <li>input_location - the folder location of the original input image</li>
  <li>image_id - the ID of the image (the filename without the .tif</li>
</ul>
<p>Pre-processed multi-channel TIFF images can be processed into this training/testing/validation structure using image_arrange_train_validate_test.py using output images from image_preprocessing.py</p>



<h2>Saving the model</h2>

Model saved to a new directory named "DyymmddThhmmss" in the path defined in the code. <br> (yymmdd and hhmmss are replaced with date and time save performed)
<p>The directory contains:
<ul><li>Model.pt - the model itself</li> 
<li>classification_class.py - the code defining the model architecture</li>
<li>settings.txt - the settings used to training the model</li>
<li>results.csv - loss and accuracy results through epochs during training</li>

<h2>References</h2>
