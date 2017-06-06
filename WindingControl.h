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
Mat ImageRequestSingle(Device* pDev);
Mat InitializeImage(Device* pDev);
void SaveSingleImageToFile(Device* pDev, string path);

// sets an chosen Output to an On or Off state
void SetOutput( Device* pDev, int Output, bool On );





// OBSOLETE //
void ProzessFrame(Device* pDev, bool stab);
void ProzessCapturedFrame(string VideoName, bool stab);
void stabilizeFrame(Mat& whole, Mat& stabframe, string method);
void stabilize(string method, const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole);
void stabilize_OpticalFlow(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole);
void stabilize_RigidTransform(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole);
void stabilize_Homography(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole);
void stabilize_Videostab(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole);
void clearBadFeatures(vector<Point2f>& feat_prev, vector<Point2f>& feat_curr, vector<uchar>& stat, vector<float>& err);
vector<KeyPoint> findFeatures(Mat& img, string method, vector<Point2f>& feat);
void drawArrows(Mat& drawframe, const vector<Point2f>& feat_1, const vector<Point2f>& feat_2);
void drawPoints(Mat& drawframe, const vector<Point2f>& feat, Scalar s);
Point2f getOpticalFlow(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr);

#endif /* defined(____WindingControl__) */