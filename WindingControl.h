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

//mvIMPACT stuff
#include <apps/Common/exampleHelper.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire.h>

using namespace std;

Device* OpenCam();
void CaptureVideoFromFile(string);
void ShowPictureOfCamera();
void WriteVideoToFile(string);


#endif /* defined(____WindingControl__) */