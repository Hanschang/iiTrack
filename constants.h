//
//  constants.h
//  
//
//  Created by Hans Chang on 2017-07-24.
//
//

#ifndef ____constants__
#define ____constants__

#include <stdio.h>
#include <string>

// Toggle for tracking methods
const bool kCalcGradient = 0;
const bool kCalcContour = 1;
const bool kCalcAverage = 0;
const bool kShowOutline = 0;

// Contant used to determine the minimum threshold value
const double kStructElementSize = 5;

// Set to true if debugging
const bool kDebugging = 0;
const bool kManualThresh = 0;

// Set to 1 for HD, Set to 0 for SD
const bool kisHighDef = 0;

// Change this to your own BioID directory
const std::string Biodir = "BioID-FaceDatabase-V1.2/";

// Directory to hold debug images
const std::string debugDir = "debug/";

// Change this to your haar cascade directory
const std::string face_cascade_name = "../../eyeTrack/haarcascade/haarcascade_frontalface_alt.xml";
const std::string eyes_cascade_name = "../../eyeTrack/haarcascade/haarcascade_eye_tree_eyeglasses.xml";

#endif /* defined(____constants__) */
