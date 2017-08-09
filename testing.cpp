#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "centerTracking.h"
#include "eyeList.h"

using namespace std;
using namespace cv;

/** Function Headers */
void display(Mat frame, Rect& face, eyeList& allEyes, bool noFace);
void findCenter(Mat frame, Rect& face, eyeList& allEyes);
bool detectEyes(Mat frame, eyeList& eyes, Rect& face);
void testFace(int imgID, int minThresh);

/** Global variables */
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
bool debugging = 1;

vector<double> means(3);
int minThresh = 50;

/** @function main */
int main( int argc, const char** argv )
{
    ofstream data("data.csv");
    data << "Average Intensity,Threshold Percentage\n";

    // Load haar cascade
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    // Trackbar for the minimum threshold value
    int slider_max = 100;
    namedWindow(window_name,1);
    string sliderName = "minThresh";
    createTrackbar( sliderName, window_name, &minThresh, slider_max);

    // Start and end ID in the BioID database
    int start = 1007;
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
        else if((char)c == 'a')
        {
            if(curr > start) curr--;
        }
        else if((char)c == 'q') minThresh--;
        else if((char)c == 'e') minThresh++;
        else if((char)c == '1')
        {
            std::cout << "Writing..." << means[0] << "," << minThresh << "\n";
            data << means[0] << "," << minThresh << "\n";
        }
        else if((char)c == '2')
        {
            std::cout << "Writing..." << means[1] << "," << minThresh << "\n";
            data << means[1] << "," << minThresh << "\n";
        }
        else if((char)c == '3')
        {
            std::cout << "Writing..." << means[2] << "," << minThresh << "\n";
            data << means[2] << "," << minThresh << "\n";
        }
        else if((char)c == 'p')
        {
            data.close();
            break;
        }
//        std::cout << c << std::endl;
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
        Mat frame_gray;
        cvtColor(frame, frame_gray, CV_BGR2GRAY);

        eyeList allEyes;
        Rect face;

        if(detectEyes(frame_gray, allEyes, face))
        {
            std::cout << "Face detected" << std::endl;

            if(allEyes.getSize() == 0)
            {
                std::cout << "No eyes detected" << std::endl;
            }
            else
            {
                findCenter(frame, face, allEyes);
            }
            display(frame, face, allEyes, 0);
        }
        else
        {
            std::cout << "Cannot detect face" << std::endl;
            display(frame, face, allEyes, 1);
        }
    }
    std::cout << "************************************" << std::endl;
    std::cout << std::endl;
    return;
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
            Rect smallEye(eyes[i].x, eyes[i].y + eyes[i].height * 0.15, eyes[i].width, 0.7 * eyes[i].height);
            Mat eyeROI = faceROI(smallEye);
            allEyes_.addEye(eyeROI, smallEye);

            //            Mat eyeROI = faceROI(eyes[i]);
            //            allEyes_.addEye(eyeROI, eyes[i]);
            if(kDebugging) imwrite(debugDir + "eye" + to_string(i) + ".jpg", eyeROI);

        }

    }
    else return 0;

    return 1;
}

void findCenter(Mat frame, Rect& face, eyeList& allEyes)
{
    for(int i = 0; i < allEyes.getSize(); i++)
    {

        Point Contourcenter(0,0);

        double radius = 0;
        contourTrack(allEyes, Contourcenter, radius, minThresh, i);

        if(Contourcenter.x > 0 || Contourcenter.y > 0)
        {

            Contourcenter.x  = Contourcenter.x + allEyes.getX(i) + face.x;
            Contourcenter.y = Contourcenter.y + allEyes.getY(i) + face.y;

            std::cout << "C: (" << Contourcenter.x << "," << Contourcenter.y << ")" << std::endl;

            circle(frame, Contourcenter, 1, Scalar(0,0,255), 1, 8 ,0);
            if(kShowOutline)
            {
                circle(frame, Contourcenter, radius, Scalar(0,0,255), 1, 8, 0);
            }
        }
        else
        {
            std::cout << "C: Cannot find" << std::endl;
        }
    }
}


// Function display
void display(Mat frame, Rect& face, eyeList& allEyes, bool noFace)
{
    Mat concatImage(45, 384, CV_8UC3, Scalar(0,0,0));

    string text = to_string(minThresh) + "%";
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
            int width = 384;
            resize(processImage, processImage, Size(183, 35));

            if(i > 1) continue;
            
            Rect destROI(Point(width / 2 * i + 6, 5), processImage.size());
            processImage.copyTo(concatImage(destROI));
        }
    }
    
    vconcat(frame, concatImage, frame);
    // show the frame
    imshow(window_name, frame);
}