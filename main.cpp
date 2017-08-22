#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include "centerTracking.h"
#include "constants.h"
#include "eyeList.h"

using namespace std;
using namespace cv;

/** Function Headers */
void display(Mat frame, Rect& face, eyeList& allEyes, bool noFace);
void findCenter(Mat frame, Rect& face, eyeList& allEyes);
bool detectEyes(Mat frame, eyeList& eyes, Rect& face);
void resize(Mat& input);

/** Global variables */
// Cascade variables declared in main function, and used in detectEyes
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
// String containing name of the main window, ensure all images are displayed together
string window_name = "Capture - Face detection";
// Trackbar position that will be used for thresholding
int trackbarPos = 20;
// Constant for video stream
bool vidWrite = 0;
// Constant for whether the video is paused
bool stopped = 0;

/** @function main */
int main( int argc, const char** argv )
{

    // Initialize videocapture and the frame it reads into
    VideoCapture capture(0);
    Mat frame;

    // Load haar cascade, quit if cannot be loaded
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    // Create the main window, and add trackbar for threshold
    namedWindow(window_name,1);
    createTrackbar("Adjustment: ", window_name, &trackbarPos, 40);

    // Switch between HD and SD
    if(!kisHighDef)
    {
        capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    }

    // Quit if camera cannot be opened
    if( ! capture.isOpened() ) return -1;

    // Capture the first frame
    capture >> frame;
    Mat frameCpy;

    while( true )
    {
        // Check first if frame is empty for not, move on to the next frame if empty
        if(kDebugging) imwrite(debugDir + "capture.jpg", frame);
        if(frame.empty()) { printf(" --(!) No captured frame -- Break!");  continue;}

        // Variable used
        Mat frame_gray;
        eyeList allEyes;
        Rect face;

        // Make a copy of the frame first in case user choose to reuse it
        frame.copyTo(frameCpy);
        cvtColor(frame, frame_gray, CV_BGR2GRAY);

        // Call detect eyes function to get the face and eye regions
        //If no face is found, display the frame as is, and move on to the next frame
        if(!detectEyes(frame_gray, allEyes, face))
        {
            display(frame, face, allEyes, 1);
            waitKey(1);
            capture >> frame;
            continue;
        }

        // If face is found, then call findCenter display
        findCenter(frame, face, allEyes);
        display(frame, face, allEyes, 0);

        // Debugging mode
        if(kDebugging)
        {
            // Write the image into debugging directory
            imwrite(debugDir + "frame.jpg", frame);

            // If user presses 'c', reuse the current frame
            int c = waitKey(0);
            if((char)c == 'c')
            {
                frameCpy.copyTo(frame);
                continue;
            }
            else if((char)c == 'q') return 0;
            // If any other key, move on to the next capture
            capture >> frame;
        }
        // Normal mode
        else
        {
            // If the program is not already paused, check if the
            //user pressed 'c'. If so, pause program
            if(!stopped)
            {
                int c = waitKey(1);
                if((char)c == 'c') stopped = 1;
                else if ((char)c == 'w') vidWrite = !vidWrite;
                else if ((char)c == 'q') return 0;
            }

            // If program is paused, stay paused as long as user keep
            //pressing 'c'. Move on if any other key is pressed
            if(stopped)
            {
                int c = waitKey(0);
                if((char)c == 'c')
                {
                    frameCpy.copyTo(frame);
                    continue;
                }
                else if ((char)c == 'w') vidWrite = !vidWrite;
                else if ((char)c == 'q') return 0;
            }
            capture >> frame;
            stopped = 0;
        }

        //Add escape for infinite loop
    }

    return 0;
}

// Takes in a grayscale frame, an eyeList container, and a Rect.
// The function should return 0 if no face is detected with face_cascade
// If face is detected, the function will store the face region in face_,
//and all detected eye regions ito the eyeList container.
bool detectEyes(Mat frame_gray_, eyeList& allEyes_, Rect& face_)
{
    vector<Rect> faces;
    vector<Rect> eyes;

    // Use haar cascade to find the biggest face in the frame
    face_cascade.detectMultiScale( frame_gray_, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
    // If face is found
    if(faces.size() > 0)
    {
        // Save the first (and only) face region in face_
        face_ = faces[0];
        Mat faceROI = frame_gray_(faces[0]);
        if(kDebugging) imwrite(debugDir + "face.jpg", faceROI);

        // Apply Gaussian blur to smooth out the image
        //The 0.005 comes from Tristum's blog (check README for reference)
        GaussianBlur(faceROI, faceROI, Size(0,0), faceROI.cols * 0.005);
        if(kDebugging) imwrite(debugDir + "faceGaussian.jpg", faceROI);

        // Detect all the eyes in the face region again using haar cascade
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(40, 40) );

        // Loop through all the detected eyes
        for(int i = 0; i < eyes.size(); i++)
        {
            // If the top left corner is too far down the face, ignore it,
            //since it's probably nostril
            if(eyes[i].y > faceROI.rows * 0.4) continue;

            // Create a smaller rectangle for more accurate eye center detection
            //Crop the top and bottom 15% of the eye region
            Rect smallEye(eyes[i].x, eyes[i].y + eyes[i].height * 0.15, eyes[i].width, 0.7 * eyes[i].height);
            Mat eyeROI = faceROI(smallEye);
            // Save the eye region into the eyeList container
            allEyes_.addEye(eyeROI, smallEye);

            if(kDebugging) imwrite(debugDir + "eye" + to_string(i) + ".jpg", eyeROI);

        }

    }
    // If no face is detected, return 0
    else return 0;

    return 1;
}


// Takes in the original colored frame, a Rect containing the face region, and
//all the eye regions in a single eyeList container
// The function calculates the eye centers for all the eye regions using one of three
//different methods: gradient, contour, and average of two. It will display the calculated
//eye centers on the frame
void findCenter(Mat frame, Rect& face, eyeList& allEyes)
{
    // Loop through each eye region in the container
    for(int i = 0; i < allEyes.getSize(); i++)
    {
        // If average or gradient method is selected, call gradientTrack
        Point gradientCenter;
        if(kCalcGradient | kCalcAverage)
        {
            Mat eyeROI = allEyes.getROI(i);
            Point center = gradientTrack(eyeROI);
            gradientCenter = Point(face.x + allEyes.getX(i) + center.x, face.y + allEyes.getY(i) + center.y);
            circle(frame, gradientCenter, 1.5, Scalar(0, 255, 0), 1, 8, 0);
        }

        // If contour method is selected, call contourTrack
        Point Contourcenter(0,0);
        if(kCalcContour | kCalcAverage)
        {
            double radius = 0;
            contourTrack(allEyes, Contourcenter, radius, 30 + trackbarPos, i);

            // If a center is detected, then display onto the frame
            if(Contourcenter.x > 0 || Contourcenter.y > 0)
            {

                Contourcenter.x  = Contourcenter.x + allEyes.getX(i) + face.x;
                Contourcenter.y = Contourcenter.y + allEyes.getY(i) + face.y;
                circle(frame, Contourcenter, 1, Scalar(0,0,255), 1, 8 ,0);

                // If user wants to show the outline of the minimal enclosing circle
                if(kShowOutline)
                {
                    circle(frame, Contourcenter, radius, Scalar(0,0,255), 1, 8, 0);
                }
            }
        }

        // Average (between contour and gradient) is selected, find the average
        //x and y value, and display onto the screen
        if(kCalcAverage)
        {
            Point avgCenter(gradientCenter.x, gradientCenter.y);
            if(Contourcenter.x > 0 || Contourcenter.y > 0)
            {
                avgCenter.x = (gradientCenter.x + Contourcenter.x) / 2;
                avgCenter.y= (gradientCenter.y + Contourcenter.y) / 2;
            }
            circle(frame, avgCenter, 1, Scalar(0,255,255), 1, 8 ,0);
        }
    }

}


// Takes in the original colored frame, face region, eyeList container,
//as well as a boolean of whether the face region is detected or not
// The function draws all the face and eye regions detected in a blue outline,
//and display the processing images on the bottom of the window
void display(Mat frame, Rect& face, eyeList& allEyes, bool noFace)
{
    // Create a solid black Mat depending on the resolution
    Mat concatImage(1, 1, CV_8UC3, Scalar(0,0,0));
    if (kisHighDef) resize(concatImage, concatImage, Size(1080,102));
    else resize(concatImage, concatImage, Size(640,60));

    // Put the threshold percentage on the top left corner of the screen
    string text = to_string(30 + trackbarPos) + "%";
    Size textSize = getTextSize(text, FONT_HERSHEY_PLAIN, 1, 1, NULL);
    rectangle(frame, Point(8,22), Point(12 + textSize.width, 18 - textSize.height), Scalar(0), CV_FILLED);
    putText(frame, text, Point(10,20), FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));

    if(stopped)
    {
        int x = frame.cols;
        int y = frame.rows;
        rectangle(frame, Point(x- 5, y - 5), Point(x - 10, y - 15), Scalar(0,0,255), CV_FILLED);
        rectangle(frame, Point(x - 15, y - 5), Point(x - 20, y - 15), Scalar(0,0,255), CV_FILLED);
    }

    // If no face is detected, just add the previously created black Mat
    //on to the bottom of the screen and display, skip the rest since there's
    //no eye region to process
    if(noFace)
    {
        vconcat(frame, concatImage, frame);
        imshow(window_name, frame);
        return;
    }

    // Create the bottom left and top right corder of the face, then draw a rectangle
    //using those coordinates for the face
    Point BLFace(face.x, face.y);
    Point TRFace(face.x + face.width, face.y + face.height);
    rectangle(frame, BLFace, TRFace, Scalar(255,0,0), 1, 8, 0);

    // Go through each eye, again finding the bottom left and top right, draw rectangle
    for(int i = 0; i < allEyes.getSize(); i++)
    {
        // Make sure only 2 eye regions are displayed
        if(i > 1) break;

        // Get the BL and TR corner of each eye, and draw rectangle
        Point BLeye(face.x + allEyes.getX(i), face.y + allEyes.getY(i));
        Point TReye(face.x + allEyes.getX(i) + allEyes.getWidth(i), face.y + allEyes.getY(i) + allEyes.getHeight(i));
        rectangle(frame, BLeye, TReye, Scalar(255,0,0), 1, 8, 0);

        // Get the processing images of the eye
        Mat processImage = allEyes.getProcessImage(i);
        cvtColor(processImage, processImage, CV_GRAY2BGR);

        // If the image is not empty
        if(! processImage.empty())
        {
            //resize the image to approx half the width of the frame
            int width;
            if(kisHighDef)
            {
                resize(processImage, processImage, Size(631, 97));
                width = 1080;
            }
            else{
                resize(processImage, processImage, Size(311, 48));
                width = 640;
            }

            // paste the processing image on to the black Mat created before
            Rect destROI(Point(width / 2 * i + 6, 6), processImage.size());
            processImage.copyTo(concatImage(destROI));
        }
    }
    // Concatenate the resulting image to the original frame, and display
    vconcat(frame, concatImage, frame);
    imshow(window_name, frame);
}


