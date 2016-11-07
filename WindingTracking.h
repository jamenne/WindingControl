//  WindingControl.h
//
// Copyright 2015 TU Dortmund, Physik, Experimentelle Physik 5
//
// Author: Janine Müller
//
//
//

#ifndef ____WindingTest__
#define ____WindingTest__

#include <string>

//OpenCV stuff
#include <opencv2/highgui/highgui.hpp>  // GUI windows and trackbars
#include <opencv2/video/video.hpp>  // camera/video classes, imports background_segm
#include <opencv2/calib3d/calib3d.hpp>  // for FindHomography
#include <opencv2/features2d/features2d.hpp>  // for FeatureDetection
#include <opencv2/videostab/stabilizer.hpp>
#include <opencv2/nonfree/nonfree.hpp>


using namespace std;
using namespace cv;

void ChooseRect(int event, int x, int y, int flags, void* output);

void TestCallBack();

void ProzessCapturedFrame2(string VideoName);

#endif /* defined(____WindingTest__) */