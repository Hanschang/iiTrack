#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "tracking.h"

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame);
void display(Mat frame, Rect face, vector<Rect> eyes);
void testFace(int imgID, int minThresh);

/** Global variables */
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
bool debugging = 1;

/** @function main */
int main( int argc, const char** argv )
{
    // Load haar cascade
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    // Trackbar for the minimum threshold value
    int minThresh;
    int slider_max = 100;
    namedWindow(window_name,1);
    string sliderName = "minThresh";
    createTrackbar( sliderName, window_name, &minThresh, slider_max);

    // Start and end ID in the BioID database
    int start = 1498;
    int end = 1600;
    int curr = start;

    // "D" for next, "A" for previous
    while(1)
    {
        testFace(curr, minThresh);

        int c = waitKey(0);
        if((char)c == 'd')
        {
            if(curr < end) curr++;
        }
        if((char)c == 'a')
        {
            if(curr > start) curr--;
        }
    }

    return 0;
}


//int main( int argc, const char** argv )
void testFace(int imgID, int minThresh)
{
    std::cout << "************************************" << std::endl;
    std::cout << "Testing ID: " << imgID << std::endl;
    std::cout << "Minimum Threshold: " << minThresh << std::endl;

    // Initialize videocapture and the frame it reads into
//    VideoCapture capture(0);
    string fileName = Biodir + "BioID_" + to_string(imgID);
    Mat frame = imread(fileName + ".pgm");

    string buffer;
    int LX, LY, RX, RY;
    fstream ifile(fileName  + ".eye");
    if(ifile.is_open())
    {
        getline(ifile, buffer);
        getline(ifile, buffer);
        stringstream ss(buffer);
        ss >> LX >> LY >> RX >> RY;
        std::cout << "L: (" << LX << "," << LY << ")  R: (" << RX << "," << RY << ")" << std::endl;
        ifile.close();
    }
    else{
        std::cout << "cannot open .eye file" << std::endl;
    }

    if( !frame.empty() )
    {
//        std::cout << "Got here" << std::endl;
        std::vector<cv::Rect> faces;
        std::vector<Rect> eyes;
        std::vector<cv::Mat> rgbChannels(3);
        cv::split(frame, rgbChannels);
        cv::Mat frame_gray = rgbChannels[2];

        // Use haar cascade to detect potential faces
        face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(40, 40) );


        Mat faceROI;
        vector<Rect> actualEyes;
        if(faces.size() > 0)
        {
            std::cout << "face Detected" << std::endl;
            // Use the first face, apply gaussian blur to reduce noise, then detect eyes
            faceROI = frame_gray(faces[0]);
            GaussianBlur(faceROI, faceROI, Size(0,0), faceROI.cols * 0.005);
            eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(15, 15) );

            std::vector<Rect> smallerEyes;
            // Go through potential eyes
            if(eyes.size() == 0)
            {
                std::cout << "No eyes detected" << std::endl;
            }
            for(int i = 0; i < eyes.size(); i++)
            {

                // If the top left corner of the "eye" is below 0.4 of the face, then it is definitely
                //not an eye, remove
                if(eyes[i].y > faceROI.rows * 0.4)
                {
                    continue;
                }

                // Create a smaller rectangle for more accurate eye center detection

                Rect smallEye(eyes[i].x + eyes[i].width * 0.10, eyes[i].y + eyes[i].height * 0.10, 0.8 * eyes[i].width, 0.8 * eyes[i].height);
                Mat eyeROI = faceROI(smallEye);
                eyes[i] = smallEye;
                actualEyes.push_back(eyes[i]);

                Scalar mean = cv::mean(eyeROI);
                std::cout << "Average: " << mean[0] << "\t";

                Point Contourcenter;
                double radius = 0;
                houghTrack(eyeROI, Contourcenter, radius, minThresh, 1, i);

                if(Contourcenter.x > 0 || Contourcenter.y > 0)
                {
                    Point adjustedCenter(Contourcenter.x + eyes[i].x + faces[0].x, Contourcenter.y + eyes[i].y + faces[0].y);
                    std::cout << "C: (" << adjustedCenter.x << "," << adjustedCenter.y << ")" << std::endl;
                    //                        circle(frame, adjustedCenter, radius, Scalar(0,255,0), 1, 8 ,0);
                    circle(frame, adjustedCenter, 1, Scalar(0,0,255), 1, 8 ,0);
                }
                else{
                    std::cout << "C: Cannot find" << std::endl;
                }
            }
        }
        // Display the detected face, eyes, and eye center
        if(!faces.empty())
        {
            display(frame, faces[0], actualEyes);
        }
        else {
            std::cout << "No face detected" << std::endl;
            imshow(window_name, frame);
        }
        std::cout << "************************************" << std::endl;
        std::cout << std::endl;
    }

    return;
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