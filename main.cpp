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
bool detectEyes(Mat frame, eyeList& eyes, Rect& face);
void resize(Mat& input);

/** Global variables */
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
int trackbarPos = 15;

/** @function main */
int main( int argc, const char** argv )
{

    // Initialize videocapture and the frame it reads into
    VideoCapture capture(0);
    Mat frame;

    // Load haar cascade
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    namedWindow(window_name,1);
    createTrackbar("Adjustment: ", window_name, &trackbarPos, 30);

    // For 480p, comment out the below lines for 720p video
    //(will slow down the program)
    if(!kisHighDef)
    {
        capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    }

    // If camera opened successfully
    if( ! capture.isOpened() ) return -1;

    Mat copyFrame;
    bool reCalc = 0;
    while( true )
    {
        // Capture the frame
        if(reCalc) copyFrame.copyTo(frame);
        else capture >> frame;
        frame.copyTo(copyFrame);
        reCalc = 0;
        if(kDebugging) imwrite(debugDir + "capture.jpg", frame);

        if(frame.empty()) { printf(" --(!) No captured frame -- Break!");  continue;}

        Mat frame_gray;
        eyeList allEyes;
        Rect face;

        cvtColor(frame, frame_gray, CV_BGR2GRAY);

        if(!detectEyes(frame_gray, allEyes, face))
        {
            display(frame, face, allEyes, 1);
            waitKey(1);
            continue;
        }

        for(int i = 0; i < allEyes.getSize(); i++)
        {
            Point gradientCenter;
            if(kCalcGradient | kCalcAverage)
            {
                Mat eyeROI = allEyes.getROI(i);
                Point center = gradientTrack(eyeROI);
                gradientCenter = Point(face.x + allEyes.getX(i) + center.x, face.y + allEyes.getY(i) + center.y);
                circle(frame, gradientCenter, 1.5, Scalar(0, 255, 0), 1, 8, 0);
            }
            
            Point Contourcenter(0,0);
            if(kCalcContour | kCalcAverage)
            {
                double radius = 0;
                contourTrack(allEyes, Contourcenter, radius, 45 + trackbarPos, i);

                if(Contourcenter.x > 0 || Contourcenter.y > 0)
                {

                    Contourcenter.x  = Contourcenter.x + allEyes.getX(i) + face.x;
                    Contourcenter.y = Contourcenter.y + allEyes.getY(i) + face.y;

                    circle(frame, Contourcenter, 1, Scalar(0,0,255), 1, 8 ,0);
                    if(kShowOutline)
                    {
                        circle(frame, Contourcenter, radius, Scalar(0,0,255), 1, 8, 0);
                    }
                }
            }

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

        display(frame, face, allEyes, 0);

        // Pause or continue with capture depending on mode
        if(kDebugging)
        {
            imwrite(debugDir + "frame.jpg", frame);
            int c = waitKey(0);
            if((char)c == 'c') reCalc = 1;
        }
        else
        {
            int c = waitKey(1);
            if((char)c == 'c') waitKey(0);
        }
    }

    return 0;
}

bool detectEyes(Mat frame_gray_, eyeList& allEyes_, Rect& face_)
{
    vector<Rect> faces;
    vector<Rect> eyes;
    face_cascade.detectMultiScale( frame_gray_, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
    if(faces.size() > 0)
    {
        face_ = faces[0];
        Mat faceROI = frame_gray_(faces[0]);
        if(kDebugging) imwrite(debugDir + "face.jpg", faceROI);


        GaussianBlur(faceROI, faceROI, Size(0,0), faceROI.cols * 0.005);
        if(kDebugging) imwrite(debugDir + "faceGaussian.jpg", faceROI);

        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(40, 40) );

        for(int i = 0; i < eyes.size(); i++)
        {
            if(eyes[i].y > faceROI.rows * 0.4) continue;
//            if(i > 1) break;

            // Create a smaller rectangle for more accurate eye center detection
            Rect smallEye(eyes[i].x + eyes[i].width * 0.05, eyes[i].y + eyes[i].height * 0.15, 0.9 * eyes[i].width, 0.7 * eyes[i].height);
            Mat eyeROI = faceROI(smallEye);
            allEyes_.addEye(eyeROI, smallEye);
            if(kDebugging) imwrite(debugDir + "eye" + to_string(i) + ".jpg", eyeROI);

        }

    }
    else return 0;

    return 1;
}


// Function display
void display(Mat frame, Rect& face, eyeList& allEyes, bool noFace)
{
    Mat concatImage(1, 1, CV_8UC3, Scalar(0,0,0));
    if (kisHighDef) resize(concatImage, concatImage, Size(1080,102));
    else resize(concatImage, concatImage, Size(640,60));

    string text = to_string(45 + trackbarPos) + "%";
    Size textSize = getTextSize(text, FONT_HERSHEY_PLAIN, 1, 1, NULL);
    rectangle(frame, Point(8,22), Point(12 + textSize.width, 18 - textSize.height), Scalar(0), CV_FILLED);
    putText(frame, text, Point(10,20), FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));

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
        if(i > 1) break;

        Point BLeye(face.x + allEyes.getX(i), face.y + allEyes.getY(i));
        Point TReye(face.x + allEyes.getX(i) + allEyes.getWidth(i), face.y + allEyes.getY(i) + allEyes.getHeight(i));
        rectangle(frame, BLeye, TReye, Scalar(255,0,0), 1, 8, 0);

        Mat processImage = allEyes.getProcessImage(i);
        if(processImage.empty()) continue;

        cvtColor(processImage, processImage, CV_GRAY2BGR);

        if(! processImage.empty())
        {
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


            if(i > 1) continue;

            Rect destROI(Point(width / 2 * i + 6, 6), processImage.size());
            processImage.copyTo(concatImage(destROI));
        }
    }

    vconcat(frame, concatImage, frame);
    // show the frame
    imshow(window_name, frame);
}


