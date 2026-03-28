
startingvalue=155
n = nImages;
images = newArray(n);
for (i=0; i < n; i++)
{
	selectImage(i+1);
	images[i]=getTitle();
	
}

for (i=0; i < images.length; i++)
{
	selectImage(images[i]);
	run("Z Project...", "projection=[Max Intensity]");
	intensity=getTitle();
	run("Size...", "width=256 height=256 depth=4 constrain average interpolation=Bilinear");
	run("Arrange Channels...", "new=24");
	saveAs("Tiff", "C:/Users/jason/Desktop/"+i+startingvalue+".tif");
	close(intensity);
	close(images[i]);
	close(i+startingvalue+".tif");
}
