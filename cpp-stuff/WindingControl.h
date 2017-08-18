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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//mvIMPACT stuff
#include <apps/Common/exampleHelper.h>
#include <common/minmax.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire.h>


using namespace std;
using namespace cv;

void CaptureVideoFromFile(string);
void WriteVideoToFile(string, Device*);
Device* OpenCam(DeviceManager &devMgr);
void ShowPictureOfCamera(Device* pDev);
Mat ImageRequestSingle(Device* pDev);
Mat InitializeImage(Device* pDev);
void SaveSingleImageToFile(Device* pDev, string path);

// sets an chosen Output to an On or Off state
void SetOutput( Device* pDev, int Output, bool On );


#endif /* defined(____WindingControl__) */