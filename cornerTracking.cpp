//
//  cornerTracking.cpp
//  
//
//  Created by Hans Chang on 2017-07-25.
//
//

#include "cornerTracking.h"

void cornerTrack(Mat eyeROI)
{
//    int thresh = 230;
    Scalar eyeMean = mean(eyeROI);
    int thresh = eyeMean[0] * 2.5;
    std::cout << thresh << std::endl;

    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( eyeROI.size(), CV_32FC1 );

    cornerHarris( eyeROI, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );

    for( int j = 0; j < dst_norm.rows ; j++ )
    { for( int i = 0; i < dst_norm.cols; i++ )
    {
        if( (int) dst_norm.at<float>(j,i) > thresh )
        {
            circle( eyeROI, Point( i, j ), 1,  Scalar(255), 1, 8, 0 );
        }
    }
    }

    imshow("eyeROI", eyeROI);

}