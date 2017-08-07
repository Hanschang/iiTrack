# eyeTrack

An OpenCV project that uses webcam to detect and track eye movement.

The program uses two different methods to detect the center of the eye: gradient and contour. Future goals of this project include tracking the corners and edges of the eye in order to track eye gaze. 

## Gradient method

Using [Fabian Timm's paper](http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=7153202) as well as help from [Tristan Hume's blog](http://thume.ca/projects/2012/11/04/simple-accurate-eye-center-tracking-in-opencv/) I was able to track eye centers using the gradient of the eye region

This method is accurate under good lighting conditions, but can get jittery when the lighting's too bright or dark.

## Contour method

Using this [IEEE publication](http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=7153202) as well as [Christian Rathgeb's paper](https://books.google.com.tw/books?hl=en&lr=&id=JVxDAAAAQBAJ&oi=fnd&pg=PR3&dq=From+Segmentation+to+Template+Security&ots=AUYgFY8yzA&sig=qUikdIDJZuQKP6muLCbXErNrBJ4&redir_esc=y#v=onepage&q=From%20Segmentation%20to%20Template%20Security&f=false) as reference, I was able to accurate track eye movement by finding the contours around the iris.

This method first use threshold to isolate the darkest parts of the eye (the iris), use morphological opening and closing to smooth out the iris regino, calculate the contour around the region, then find the minimally enclosing circle as well as its center. 

Through testing, I found this method to be more stable and accurate than the Gradient method. However, it still has difficulty finding the eye center under extreme conditions such as very bright or very dark settings.

## Main.cpp
Opens up the default webcam on the computer/laptop. You can toggle different tracking methods (gradient, contour, or the average of the two), set file directories, and toggle debugging mode, in constants.h

The program shows the detected face region, eye region, and eye centers in the frame, as well as the processing images on the bottom. The six segments in the processing image shows the 6 stages used in the contour method: (1)original, (2)threshold, (3)morphological operations, (4)calculate contour, (5)calculate enclosing circle, (6)find circle center.

Use the trackbar on the top to adjust the threshold percentage. The program calculates the minimum threshold value by first finding the average intensity around each eye region, then multiply the average by the threshold percentage set by the user. In a darker setting, lower threshold percentage should be used, and vice versa. 

__normal mode: __ Continuously capture and process frames. You  can pause anytime by pressing the key 'c', and any key to unpause.

__debugging mode: __ Only captures single frame at a time. You  can move onto the next frame by pressing any key other than 'c'. Pressing 'c' will save all debugging images into the '/debug' directory (Can be changed in constants.h). It can also recalculate the eye centers if the user wants to test different threshold values on the same frame.

## Testing.cpp

Use testing.cpp to test the program using the BioID database. You can download the database [here](https://www.bioid.com/About/BioID-Face-Database). Use the A and D key and go to the previous or next photo in the database, and use the trackbar above to manually set the minimum threshhold value for the contour method. Use 'Q' and 'E' to cycle through the database.

The eye coordinates provided by BioID, as well as the coordinates calculated by the program are printed for each image. You can use this information to calculate the accuracy and find the optimal threshold value. Pressing '1' or '2' writes the coordinates for the respective eye into a xml file for recording purposes. 