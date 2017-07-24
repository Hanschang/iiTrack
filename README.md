## eyeTrack

An OpenCV project that uses webcam to detect and track eye movement.

The program uses two different methods to detect the center of the eye: gradient and contour. Future goals of this project include tracking the corners and edges of the eye in order to track eye gaze. 

## Gradient method

using [Fabian Timm's paper](http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=7153202) as well as help from [Tristan Hume's blog](http://thume.ca/projects/2012/11/04/simple-accurate-eye-center-tracking-in-opencv/) I was able to track eye centers using the gradient of the eye region

This method is accurate under good lighting conditions, but can get jittery when the lighting's too bright or dark.

## Contour method

using this [IEEE publication](http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=7153202) as well as [Christian Rathgeb's paper](https://books.google.com.tw/books?hl=en&lr=&id=JVxDAAAAQBAJ&oi=fnd&pg=PR3&dq=From+Segmentation+to+Template+Security&ots=AUYgFY8yzA&sig=qUikdIDJZuQKP6muLCbXErNrBJ4&redir_esc=y#v=onepage&q=From%20Segmentation%20to%20Template%20Security&f=false) as reference, I was able to accurate track eye movement by finding the contours around the iris.

This method first use threshold to isolate the darkest parts of the eye (the iris), find the contour around the region, and find the minimally enclosing circle as well as its center. 

Through testing, I found this method to be more stable and accurate than the Gradient method, however it might be unable to find the center under very dark lighting

## Testing.cpp

Use testing.cpp to test the program using the BioID database. You can download the database [here](https://www.bioid.com/About/BioID-Face-Database). Use the A and D key and go to the previous or next photo in the database, and use the trackbar above to manually set the minimum threshhold value for the contour method. 