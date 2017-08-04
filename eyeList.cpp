//
//  eyeList.cpp
//  
//
//  Created by Hans Chang on 2017-08-04.
//
//

#include "eyeList.h"

eyeList::eyeList() : size_(0) { };

eyeList::eyeList(Mat ROI, Rect Region) : size_(1)
{
    // Push the ROI and Rect into their respective vectors
    //and push an empty Mat to processImage to be replaced later
    eyeRegion_.push_back(Region);
    ROI_.push_back(ROI);
    processImages_.push_back(Mat());
}

eyeList::~eyeList() { };

void eyeList::addEye(Mat ROI, Rect eyeRegion)
{
    // Push the ROI and Rect into their respective vectors
    //and push an empty Mat to processImage to be replaced later
    eyeRegion_.push_back(eyeRegion);
    ROI_.push_back(ROI);
    processImages_.push_back(Mat());
    size_++;
}

void eyeList::addProcessImage(Mat processImage, int index)
{
    if (index >= size_ || index < 0) return;

    processImages_.push_back(processImage);
}

int eyeList::getX(int index)
{
    if (index >= size_ || index < 0) return -1;

    Rect eye = eyeRegion_[index];
    return eye.x;
}

int eyeList::getY(int index)
{
    if (index >= size_ || index < 0) return -1;

    Rect eye = eyeRegion_[index];
    return eye.y;

}

Mat eyeList::getROI(int index)
{
    if (index >= size_ || index < 0) return Mat();
    return ROI_[index];
}

Mat eyeList::getProcessImage(int index)
{
    if (index >= size_ || index < 0) return Mat();
    return processImages_[index];
}

int eyeList::getSize()
{
    return size_;
}




