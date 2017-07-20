//
//  tracking.cpp
//  
//
//  Created by Hans Chang on 2017-07-13.
//
//

#include "tracking.h"
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
//            vector<double> displacement  = findDispVec(i, j, x, y);
//            vector<double> displacement(2);

//            vector<double> disp(2);

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

Point trackEyeCenter(Mat eyeROI)
//Point trackEyeCenter(Mat eyeROI)
{
    // Calculate the x and y gradient
    Mat xGrad = calcGradient(eyeROI,0);
    Mat yGrad = calcGradient(eyeROI.t(), 1);

    // Debugging purpose
//    imshow("xGrad", xGrad);
//    imshow("yGrad", yGrad);

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
//    imshow("normXGrad", xGrad);
//    imshow("normYGrad", yGrad);

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

//    Point center = minMaxLoc(result, NULL, NULL, NULL, center)
    Point center;
    minMaxLoc(result, NULL, NULL, NULL, &center);
//    std::cout << "Center: " << center.x << "," << center.y << std::endl;


    return center;
}








