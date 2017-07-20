#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "tracking.h"

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame);
void display(Mat frame, Rect face, vector<Rect> eyes);

/** Global variables */
String face_cascade_name = "../../eyeTrack/haarcascade/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "../../eyeTrack/haarcascade/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
bool debugging = 1;
string imgName = "";

/** @function main */
int main( int argc, const char** argv )
{

    // Initialize videocapture and the frame it reads into
    VideoCapture capture(0);
    Mat frame;

    // Load haar cascade
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    // For 480p, comment out the below lines for 720p video
    //(will slow down the program)
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    // If camera opened successfully
    if( capture.isOpened() )
    {
        while( true )
        {
            // Capture the frame
            capture >> frame;

            if( !frame.empty() )
            {
                std::vector<cv::Rect> faces;
                std::vector<Rect> eyes;
                std::vector<cv::Mat> rgbChannels(3);
                cv::split(frame, rgbChannels);
                cv::Mat frame_gray = rgbChannels[2];

                // Use haar cascade to detect potential faces
                face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );


                Mat faceROI;
                vector<Rect> actualEyes;
                if(faces.size() > 0)
                {
                    // Use the first face, apply gaussian blur to reduce noise, then detect eyes
                    faceROI = frame_gray(faces[0]);
                    GaussianBlur(faceROI, faceROI, Size(0,0), faceROI.cols * 0.005);
                    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(40, 40) );

                    std::vector<Rect> smallerEyes;

                    // Go through potential eyes
                    for(int i = 0; i < eyes.size(); i++)
                    {
                        // If the top left corner of the "eye" is below 0.4 of the face, then it is definitely
                        //not an eye, remove
                        if(eyes[i].y > faceROI.rows * 0.4)
                        {
                            continue;
                        }

                        // Create a smaller rectangle for more accurate eye center detection
                        actualEyes.push_back(eyes[i]);
                        Rect smallEye(eyes[i].x + eyes[i].width * 0.05, eyes[i].y + eyes[i].height * 0.15, 0.9 * eyes[i].width, 0.7 * eyes[i].height);
                        Mat eyeROI = faceROI(smallEye);
                        eyes[i] = smallEye;

                        // Get the center point using trackEyeCenter method, then change it to be in context of the whole frame.
                        Point center = trackEyeCenter(eyeROI);
                        Point faceCenter = Point(faces[0].x + eyes[i].x + center.x, faces[0].y + eyes[i].y + center.y);
                        circle(frame, faceCenter, 1.5, Scalar(0, 0, 250), 1, 8, 0);
                        
                    }

                }

                // Display the detected face, eyes, and eye center
                if(!faces.empty()) display(frame, faces[0], actualEyes);
                else imshow(window_name, frame);

            }
            else
            { printf(" --(!) No captured frame -- Break!"); break; }

            int c = waitKey(1);
            if( (char)c == 'c' ) { imwrite("debug.jpg", frame); }
        }
    }

    return 0;
}


// Function display
void display(Mat frame, Rect face, vector<Rect> eyes)
{
    // Create the bottom left and top right corder of the face, then draw a rectangle
    //using those coordinates for the face
    Point BLFace(face.x, face.y);
    Point TRFace(face.x + face.width, face.y + face.height);
    rectangle(frame, BLFace, TRFace, Scalar(255,0,0), 1, 8, 0);

    // Go through each eye, again finding the bottom left and top right, draw rectangle
    for(int i = 0; i < eyes.size(); i++)
    {
        Point BLeye(face.x + eyes[i].x, face.y + eyes[i].y);
        Point TReye(face.x + eyes[i].x + eyes[i].width, face.y + eyes[i].y + eyes[i].height);
        rectangle(frame, BLeye, TReye, Scalar(255,0,0), 1, 8, 0);
    }

    // show the frame
    imshow(window_name, frame);
}



/** @function detectAndDisplay */
void detectAndDisplay( Mat frame)
{
    // Create a gray version of the frame. Also keep an original version for debugging
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat debugFrame;
    frame.copyTo(debugFrame);
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    // Go through all the detected faces
    for( size_t i = 0; i < faces.size(); i++ )
    {
        // turn the face region into Mat
        Mat faceROI = frame_gray( faces[i] );
        Mat debugFace = debugFrame(faces[i]);

        // Get the bottom left and top right corner of the face, draw it on the frame
        Point BLFace(faces[i].x, faces[i].y);
        Point TRFace(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        rectangle(frame, BLFace, TRFace, Scalar(255,0,0), 1, 8, 0);

        // Vector to hold the eyes
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        // Go through all the detected eyes
        for( size_t j = 0; j < eyes.size(); j++ )
        {
            Point bottomLeft(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y);
            Point topRight(faces[i].x + eyes[j].x + eyes[j].width, faces[i].y + eyes[j].y + eyes[j].height);

            // Draw the rectangle around the eye, and add a point in the center
            rectangle(frame, bottomLeft, topRight, Scalar(255, 0, 0), 1, 8, 0);


            // Get the eye image using the eyes vector
            Mat grayEye = faceROI(eyes[j]);
            Mat debugEye = debugFace(eyes[j]);

            // Showing the processed image
            string eyeNum = "debugImg/" + imgName + "/eye"+ to_string(j);
            imshow(eyeNum , debugEye );
            imwrite(eyeNum + "G.jpg", grayEye);
            if(debugging) imwrite(eyeNum + ".jpg", debugEye);

            rectangle(debugFrame, bottomLeft, topRight, Scalar(255, 0, 0), 1, 8, 0);
        }
        imshow("face", debugFace);
        if(debugging) imwrite("debugImg/" + imgName + "/debugFace.jpg", debugFace);
    }

    imshow( window_name, frame );
    if(debugging) imwrite("debugImg/" + imgName +"/frame.jpg", frame);
}