//
//  cornerTracking.h
//  
//
//  Created by Hans Chang on 2017-07-25.
//
//

#ifndef ____cornerTracking__
#define ____cornerTracking__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include "constants.h"
using namespace cv;
using namespace std;

void cornerTrack(Mat eyeROI);

#endif /* defined(____cornerTracking__) */
