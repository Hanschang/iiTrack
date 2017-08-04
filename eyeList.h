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

class eyeList {

public:
    eyeList();
    eyeList(Mat ROI, Rect Region);
    ~eyeList();

    void addEye(Mat ROI, Rect eyeRegion);
    void addProcessImage(Mat processImage, int index);

    int getX(int index);
    int getY(int index);
    int getWidth(int index);
    int getHeight(int index);
    Mat getROI(int index);
    Mat getProcessImage(int index);
    int getSize();

private:
    vector<Mat> ROI_;
    vector<Rect> eyeRegion_;
    vector<Mat> processImages_;
    int size_;
};

#endif /* defined(____eyeList__) */
