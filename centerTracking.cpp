//
//  tracking.cpp
//  
//
//  Created by Hans Chang on 2017-07-13.
//
//

#include "centerTracking.h"
#include "util.h"


Mat calcGradient(const Mat &input, const bool isVertical)
{
    // Initialize empty output Mat with the same dimension as input
    Mat gradient(input.rows, input.cols, CV_64F);


    for(int i = 0; i < input.rows; ++i)
    {
        const uchar *inputRow = input.ptr<uchar>(i);
        double *gradientRow = gradient.ptr<double>(i);

        gradientRow[0] = inputRow[1] - inputRow[0];
        for(int j = 1; j < input.cols - 1; ++j)
        {
            gradientRow[j] = (inputRow[j+1] - inputRow[j-1]) / 2.0;
        }
        gradientRow[input.cols - 1] = inputRow[input.cols-1] - inputRow[input.cols-2];
    }


    if(isVertical) return gradient.t();
    return gradient;
}

Mat calcMag(const Mat &xGrad, const Mat &yGrad)
{
    Mat magnitude(xGrad.rows, xGrad.cols, CV_64F);

    for(int i = 0; i < xGrad.rows; i++)
    {
        const double *xRow = xGrad.ptr<double>(i);
        const double *yRow = yGrad.ptr<double>(i);
        double *magRow = magnitude.ptr<double>(i);

        for(int j = 0; j < xGrad.cols; j++)
        {
            magRow[j] = sqrt(pow(xRow[j], 2) + pow(yRow[j], 2));
        }

    }

    return magnitude;
}

void calcDif(int x, int y, const cv::Mat &weight, double gradX, double gradY, Mat &output)
{

    for(int i = 0; i < output.rows; i++)
    {
        double *Ro = output.ptr<double>(i);
        const unsigned char *Wr = weight.ptr<unsigned char>(i);
        for(int j = 0; j < output.cols; j++)
        {
            if(i == x && j == y) continue;

            double displacementX = x - i;
            double displacementY = y - j;

            //    double mag = sqrt(pow(disp[0], 2) + pow(disp[1], 2));
            double mag = sqrt(displacementX * displacementX + displacementY * displacementY);
            displacementX = displacementX / mag;
            displacementY = displacementY / mag;

            double dotProd = displacementX * gradX + displacementY * gradY;

            Ro[j] += dotProd * dotProd * (Wr[j]/1.0);
        }
    }
}

vector<double> findDispVec(int x0, int y0, int x1, int y1)
{
    vector<double> disp(2);

    disp[0] = x1 - x0;
    disp[1] = y1 - y0;

//    double mag = sqrt(pow(disp[0], 2) + pow(disp[1], 2));
    double mag = sqrt(disp[0] * disp[0] + disp[1] * disp[1]);
    disp[0] = disp[0] / mag;
    disp[1] = disp[1] / mag;

    return disp;
}

Point gradientTrack(Mat eyeROI)
//Point trackEyeCenter(Mat eyeROI)
{
    // Calculate the x and y gradient
    Mat xGrad = calcGradient(eyeROI,0);
    Mat yGrad = calcGradient(eyeROI.t(), 1);


    // Calculate the Magnitude
    Mat magnitude = calcMag(xGrad, yGrad);
//    imshow("magnitude", magnitude);

    // Find the threshhold using the equation
    //thresh = 0.3 * stdDev + Mean
    Scalar std, mean;
    meanStdDev(magnitude, mean, std);
//    double threshhold = 0.3 * std[0] / sqrt(xGrad.cols * xGrad.rows)  + mean[0];
    double threshhold = 0.3 * std[0] + mean[0];

    // Normalize x and y grad. Set points below threshold to 0
    for(int i = 0; i < xGrad.rows; i++)
    {
        double *Rx = xGrad.ptr<double>(i);
        double *Ry = yGrad.ptr<double>(i);
        const double *Mag = magnitude.ptr<double>(i);

        for(int j = 0 ;j < xGrad.cols; j++)
        {
            if(Mag[j] > threshhold)
            {
                Rx[j] /= Mag[j];
                Ry[j] /= Mag[j];
            }
            else{
                Rx[j] = 0.0;
                Ry[j] = 0.0;
            }
        }
    }

    Mat weight;
    GaussianBlur( eyeROI, weight, cv::Size( 5, 5 ), 0, 0 );
    for (int y = 0; y < weight.rows; ++y) {
        unsigned char *row = weight.ptr<unsigned char>(y);
        for (int x = 0; x < weight.cols; ++x) {
            row[x] = (255 - row[x]);
        }
    }

    // Compare magnitud and displacement vectors
    Mat result = Mat::zeros(xGrad.rows, xGrad.cols, CV_64F);
    for(int i = 0; i < xGrad.rows; i++)
    {
        const double *Rx = xGrad.ptr<double>(i);
        const double *Ry = yGrad.ptr<double>(i);
        for(int j = 0; j < xGrad.cols; j++)
        {
            if(Rx[j] > 0 || Ry[j] > 0)
            {

                calcDif(i, j, weight, Rx[j], Ry[j], result);
            }
        }
    }

    Point center;
    minMaxLoc(result, NULL, NULL, NULL, &center);

    return center;
}

// Find the eye center with a 5 step process
// The function should find the center of the eye, and
// Save all the processing images into the eyeList container
void contourTrack(eyeList& allEyes, Point &center, double &MaxR, int minThresh, int eyeNum)
{
    // There shouldn't be more than 2 eyes being processed on any given frame
    if(eyeNum > 1) return;

    // Get the ROI
    Mat eyeROI = allEyes.getROI(eyeNum);

    // Initialize Mat that will hold all the processing images for this eye
    //as well as all intermediate images
    Mat combined;
    Mat thresh(eyeROI.cols, eyeROI.rows, CV_64F);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // Calculate the mean intensity of the eyeROI
    //set threshold value to half of that
    Scalar eyeMean = mean(eyeROI);
    int threshhold = eyeMean[0] * minThresh/100;

    // filter out all pixels greater than the threshold value
    threshold(eyeROI, thresh, threshhold, 255, THRESH_BINARY);
    // Add the resulting image into the combined Mat
    hconcat(eyeROI, thresh, combined);

    // Morphological closing and opening to get rid of holes and bumps
    Mat element = getStructuringElement( MORPH_RECT, Size(kStructElementSize, kStructElementSize) );
    morphologyEx(thresh, thresh, MORPH_CLOSE, element);
    morphologyEx(thresh, thresh, MORPH_OPEN, element);
    // Add the image to combined mat
    hconcat(combined, thresh, combined);

    // Find all the contours in the region
    findContours(thresh, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    // variables to hold center of the eye
    Point2f centre;
    float radius;
    int maxIndex = -1;
    int maxRadius = 0;
    Point maxCentre;
    // Loop through all the contours
    for(int i = 0 ; i < contours.size(); i++)
    {
        // For each contour, if its disconnected, use approxPolyDP to close it,
        //then find the minimally enclosing circle
        vector<Point> closed_contour;
        approxPolyDP(contours[i], closed_contour, 3, true);
        minEnclosingCircle(closed_contour, centre, radius);

        // If the circle is bigger than the ROI region, then it's most likely
        //not the iris, reject it
        if(radius >= 0.5 * eyeROI.cols) continue;

        // If valid circle, check to see if its the biggest, if so update the values
        if(radius > maxRadius)
        {
            maxCentre = centre;
            maxIndex = i;
            maxRadius = radius;
        }

    }
    // If no valid circles are found, then add the precessing images to eyeList container
    // and return
    if(maxIndex == -1)
    {
        Mat blackSpace(eyeROI.rows, eyeROI.cols, CV_8UC1, Scalar(0,0,0));
        hconcat(combined, blackSpace, combined);
        allEyes.addProcessImage(combined, eyeNum);
        return;
    }

    // Draw the detected contours and minimally enclosing circles, add them to
    //the procssing images
    Mat contourEye;
    eyeROI.copyTo(contourEye);
    drawContours(contourEye, contours, maxIndex, Scalar(255,255,255), 1, 8);
    hconcat(combined, contourEye, combined);
    circle(contourEye, maxCentre, maxRadius, Scalar(255,255,255), 1, 8, 0);
    hconcat(combined, contourEye, combined);
    circle(eyeROI, maxCentre, maxRadius, Scalar(255,255,255), 1, 8, 0);
    circle(eyeROI, maxCentre, 1, Scalar(255,255,255), 1, 8, 0);
    hconcat(combined, eyeROI, combined);

    // add the processing image to the eyeList contianer
    allEyes.addProcessImage(combined, eyeNum);
    if(kDebugging)
    {
        imwrite(debugDir + "processEye" + to_string(eyeNum) + ".jpg", combined);
    }

    // Set the center and radius of the minimally enclosing circle,
    //maxCentre is the detected eye center
    MaxR = maxRadius;
    center = maxCentre;

}






