//
//  eyeList.h
//  
//
//  Created by Hans Chang on 2017-08-04.
//
//

#ifndef ____eyeList__
#define ____eyeList__

#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;


// Container class that holds all the information of eye regions, including:
//the ROI as a Rect, the ROI as a Mat, the processing images,
//and the number of eye regions
class eyeList {

public:
    // Constructors and destructors
    eyeList();
    eyeList(Mat ROI, Rect Region);
    ~eyeList();

    // adding a new eye region to the container
    void addEye(Mat ROI, Rect eyeRegion);
    // Adding a processing image of a particular eye region
    void addProcessImage(Mat processImage, int index);

    // Accessors for different variables in the container class
    int getX(int index);
    int getY(int index);
    int getWidth(int index);
    int getHeight(int index);
    Mat getROI(int index);
    Mat getProcessImage(int indeax);
    int getSize();

private:
    // Vector containing the ROI (Mat and Rect) and processing image
    vector<Mat> ROI_;
    vector<Rect> eyeRegion_;
    vector<Mat> processImages_;

    // THe number of eye regions in the container
    int size_;
};

#endif /* defined(____eyeList__) */
