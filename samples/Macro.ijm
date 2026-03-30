
startingvalue=0
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
	//run("Z Project...", "projection=[Max Intensity]");
	//intensity=getTitle();
	//run("Size...", "width=256 height=256 depth=4 constrain average interpolation=Bilinear");
	//run("Arrange Channels...", "new=24");
	saveAs("png", "C:/Users/jason/Desktop/"+substring(images[i], 0, images[i].length - 4)+".png");
	//close(intensity);
	close(images[i]);
	close(substring(images[i], 0, images[i].length - 4)+".png");
}
