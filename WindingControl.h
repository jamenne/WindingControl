//  WindingControl.h
//
// Copyright 2015 TU Dortmund, Physik, Experimentelle Physik 5
//
// Author: Janine Müller
//
//
//

#ifndef ____WindingControl__
#define ____WindingControl__

#include <string>

//OpenCV stuff
#include <opencv2/highgui/highgui.hpp>  // GUI windows and trackbars
#include <opencv2/video/video.hpp>  // camera/video classes, imports background_segm
#include <opencv2/calib3d/calib3d.hpp>  // for FindHomography
#include <opencv2/features2d/features2d.hpp>  // for FeatureDetection
#include <opencv2/videostab/stabilizer.hpp>
#include <opencv2/nonfree/nonfree.hpp>

//mvIMPACT stuff
#include <apps/Common/exampleHelper.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire.h>

using namespace std;
using namespace cv;

void CaptureVideoFromFile(string);
void WriteVideoToFile(string, Device*);
Device* OpenCam(DeviceManager &devMgr);
void ShowPictureOfCamera(Device* pDev);
void ShowPictureOfCamera2(Device* pDev);
Mat ImageRequestSingle(Device* pDev, Mat &frame);
Mat InitializeImage(Device* pDev);
void ProzessFrame();
void ProzessFrame2();
void searchForMovement(Mat thresholdImage, Mat &cameraFeed);
string intToString(int);
void inplaceHorizontalMirror( void* pData, int height, size_t pitch );



#endif /* defined(____WindingControl__) */