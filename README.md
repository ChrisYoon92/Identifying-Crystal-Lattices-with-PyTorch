# Identifying-Crystal-Lattices-with-PyTorch
Identifying Crystal Lattices with PyTorch

Codes Running Note: 

Download the zip and there are four things in the zip, TestData folder, TrainingData folder, Image Classfier code, and the pth code.

TestData folder is the place to store test images, TrainingData folder is the place to store training images. Basically we have 4 test
images for each category, and below are the amount of training images for each category.

Type                          Number of training images
Cubic                                  1318
Hexagonal                              811
Monoclinic                             948
Orthorhombic                           1255
Rhombohedral                           508
Tetragonal                             1025
Triclinic                              226

Please don't worry about the crystalstruct.pth file if you just want to test the image classfier, this pth code contains the parameters 
of the trained model.

Make sure in the directory that you run this code in, you have a folder with all the things, and you'll want to modify the directory in 
the script also.

If running on Windows and you get a BrokenPipeError, try setting the num_worker of torch.utils.data.DataLoader() to 0.

Enjoy!
