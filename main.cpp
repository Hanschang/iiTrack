#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include "tracking.h"
#include "constants.h"

using namespace std;
using namespace cv;

/** Function Headers */
void display(Mat frame, Rect face, vector<Rect> eyes);

/** Global variables */
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";


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

    namedWindow(window_name,1);

    // If camera opened successfully
    if( ! capture.isOpened() ) return -1;
//    {
    while( true )
    {

        // Capture the frame
        capture >> frame;
        if(kDebugging) imwrite(debugDir + "capture.jpg", frame);

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
                if(kDebugging) imwrite(debugDir + "face.jpg", faceROI);

                GaussianBlur(faceROI, faceROI, Size(0,0), faceROI.cols * 0.005);
                if(kDebugging) imwrite(debugDir + "faceGaussian.jpg", faceROI);

                eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(40, 40) );

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
                    Rect smallEye(eyes[i].x + eyes[i].width * 0.05, eyes[i].y + eyes[i].height * 0.15, 0.9 * eyes[i].width, 0.7 * eyes[i].height);
                    Mat eyeROI = faceROI(smallEye);
                    eyes[i] = smallEye;
                    actualEyes.push_back(smallEye);

                    if(kDebugging) imwrite(debugDir + "eye" + to_string(i) + ".jpg", eyeROI);

                    // Get the center point using trackEyeCenter method, then change it to be in context of the whole frame.
                    Point faceCenter;
                    if(kCalcGradient | kCalcAverage)
                    {
                        Point center = trackEyeCenter(eyeROI);
                        faceCenter = Point(faces[0].x + eyes[i].x + center.x, faces[0].y + eyes[i].y + center.y);
                        circle(frame, faceCenter, 1.5, Scalar(0, 255, 0), 1, 8, 0);
                    }

                    Point Contourcenter(0,0);
                    if(kCalcContour | kCalcAverage)
                    {
                        double radius = 0;
                        houghTrack(eyeROI, Contourcenter, radius, NULL, 0, i);

                        if(Contourcenter.x > 0 || Contourcenter.y > 0)
                        {
                            if(kDebugging)
                            {
                                Mat coloredFace = frame(faces[0]);
                                Mat coloredEye = coloredFace(eyes[i]);
                                Mat debugEye;
                                coloredEye.copyTo(debugEye);
                                circle(debugEye, Contourcenter, 1, Scalar(0,0,255), 1, 8, 0);
                                circle(debugEye, Contourcenter, radius, Scalar(0,0,255), 1, 8, 0);
                                imwrite(debugDir + "eyeResult" + to_string(i) + ".jpg", debugEye);
                            }

                            Contourcenter.x  = Contourcenter.x + eyes[i].x + faces[0].x;
                            Contourcenter.y = Contourcenter.y + eyes[i].y + faces[0].y;
                            //                        std::cout << "Contour method: (" << adjustedCenter.x << "," << adjustedCenter.y << ")" << std::endl;
                            //                        circle(frame, adjustedCenter, radius, Scalar(0,255,0), 1, 8 ,0);
                            circle(frame, Contourcenter, 1, Scalar(0,0,255), 1, 8 ,0);
                            if(kShowOutline)
                            {
                                circle(frame, Contourcenter, radius, Scalar(0,0,255), 1, 8, 0);
                            }
                        }
                    }

                    if(kCalcAverage)
                    {
                        Point avgCenter(faceCenter.x, faceCenter.y);
                        if(Contourcenter.x > 0 || Contourcenter.y > 0)
                        {
                            avgCenter.x = (faceCenter.x + Contourcenter.x) / 2;
                            avgCenter.y= (faceCenter.y + Contourcenter.y) / 2;
                        }
                        circle(frame, avgCenter, 1, Scalar(0,255,255), 1, 8 ,0);
                    }
                }
            }
            // Display the detected face, eyes, and eye center
            if(!faces.empty()) display(frame, faces[0], actualEyes);
            else imshow(window_name, frame);

        }
        else
        { printf(" --(!) No captured frame -- Break!"); break; }

        if(kDebugging)
        {
            int c = waitKey(0);
            if((char)c == 'c') {imwrite(debugDir + "frame.jpg", frame); waitKey(0); }
        }
        else
        {
            int c = waitKey(1);
            if((char)c == 'c') waitKey(0);
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

