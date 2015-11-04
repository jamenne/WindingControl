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

void ProzessCapturedFrame(string VideoName, bool stab, bool ChooseROI);
Mat stabilizeFrame(Mat& whole, Rect ROI, string method);
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

void CallBackFunc(int event, int x, int y, int flags, void* userdata, vector<Point> Points);
void TestCallBack();


#endif /* defined(____WindingTest__) */