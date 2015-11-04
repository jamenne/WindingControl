// WindingControl.cpp
//
// Copyright 2015 TU Dortmund, Physik, Experimentelle Physik 5
//
// Author: Janine Müller
//
//
//

#include "WindingTest.h"

//OpenCV stuff
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  // GUI windows and trackbars
#include <opencv2/video/video.hpp>  // camera/video classes, imports background_segm
#include <opencv2/calib3d/calib3d.hpp>  // for FindHomography
#include <opencv2/features2d/features2d.hpp>  // for FeatureDetection
#include <opencv2/videostab/stabilizer.hpp> // for video stabilisation
#include <opencv2/nonfree/nonfree.hpp>

#include <string>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;


void ChooseRect(int event, int x, int y, int flags, void* output)
{   
    // vector to store the output Points
    vector<Point>* Out = (vector<Point>*) output;
    // needs to be static, otherwise 0 in between mouse actions (you'll lost the first corner) 
    static Point P1;
    static Point P2;
    bool clicked = false;
    if  ( event == EVENT_LBUTTONDOWN )
    {
        // Left button of mouse is clicked
        clicked = true;
        // Points are set
        P1.x = x;
        P1.y = y;
        P2.x = x;
        P2.y = y;

    }
    else if (event == EVENT_LBUTTONUP)
    {
        // Left button of mouse is released
        // second Point is set to final position
        P2.x = x;
        P2.y = y;
        clicked = false;
        // store Points to vector to transfer them outside the function
        Out->push_back(P1);
        Out->push_back(P2);
    }

    else if (event == EVENT_MOUSEMOVE)
    {
        // mouse is moving

        //always store new position to Point P2
        if (clicked)
        {
            P2.x = x;
            P2.y = y;
        }
    }
    //cout << "P1:\t" << P1.x << "\t" << P1.y << "\tP2:\t" << P2.x << "\t" << P2.y << endl;

}

void TestCallBack(){

// Read image from file 
Mat img = imread("/Users/janine/Desktop/2267049_web.jpg");

//if fail to read the image
if ( img.empty() ) 
{ 
    cout << "Error loading the image" << endl;
    exit(EXIT_FAILURE);
}

//Create a window
namedWindow("My Window", 1);

//vector to store data
vector<Point> Out;

//set the callback function for any mouse event
setMouseCallback("My Window", ChooseRect, &Out);

//show the image
imshow("My Window", img);

// Wait until user press some key
waitKey(0);

if (!Out.empty())
{
    cout << Out[0].x << "\t" << Out[0].y << "\t" << Out[1].x << "\t" << Out[1].y << endl;   
}

}


void ProzessCapturedFrame(string VideoName, bool stab, bool ChooseROI){

    VideoCapture cap; // open the video file for reading

    namedWindow("Original Video",CV_WINDOW_NORMAL); //create a window called "Original Video"
    resizeWindow("Original Video", 800, 600);
    namedWindow("Stabilized Video",CV_WINDOW_NORMAL); //create a window called "Stabilized Video"
    // resizeWindow("Stabilized Video", 800, 600);

    Mat frame;

    char key('a');
    int wait_ms(10);
    
    while(key != 27 && key != 'q'){

        cap.open(VideoName);

        if ( !cap.isOpened() ){  // if not success, exit program

         cout << "Cannot open the video file" << endl;
         exit(EXIT_FAILURE);
        }

        //check if the video has reach its last frame.
        //we add '-1' because we are reading two frames from the video at a time.
        //if this is not included, we get a memory error!
        while(cap.get(CV_CAP_PROP_POS_FRAMES)<cap.get(CV_CAP_PROP_FRAME_COUNT) && key != 27 && key != 'q'){

            bool bSuccess = cap.read(frame); // read a new frame from video
            resize(frame, frame, Size(800, 600), 0, 0, INTER_CUBIC); //resize the image
            imshow("Original Video", frame); //show the frame in "Original Video" window

            Size s = frame.size();
            int frameheight = s.height;
            int framewidth = s.width;

            // Redion of Interest
            static Rect ROI= Rect(0,0,framewidth,frameheight);
            //vector to store data
            vector<Point> Out;
            // only finds a few features if drawn
            //rectangle(frame, ROI, Scalar( 0, 255, 255 ));


            if (!bSuccess) //if not success, break loop
            {
                cout << "Cannot read the frame from video file" << endl;
                break;
            }

            if (ChooseROI){
                //set the callback function for any mouse event
                setMouseCallback("Original Video", ChooseRect, &Out);

                waitKey(0);

                ChooseROI = false;

                // ROI = Rect(Out[0], Out[1]);

                // rectangle(frame, ROI, Scalar( 0, 255, 255 ));
                  
            }

            if (!Out.empty())
            {
                //cout << Out[0].x << "\t" << Out[0].y << "\t" << Out[1].x << "\t" << Out[1].y << endl;
                ROI = Rect(Out[0], Out[1]);
                rectangle(frame, ROI, Scalar(0,255,255));  
            }

            //rectangle(frame, ROI, Scalar( 0, 255, 255 ));


            if (stab)
            {
                //Mat frame_roi=frame(ROI);
                Mat stabframe = stabilizeFrame(frame, ROI, "PYR_LK_OPTICALFLOW");
                imshow("Stabilized Video", stabframe); //show the frame in "Stabilized Video" window
            }


            key = waitKey(wait_ms);  // Capture Keyboard stroke
        }
    }

}

Mat stabilizeFrame(Mat& whole, Rect ROI, string method) {

    static vector<Point2f> features_prev;
    vector<Point2f> features_curr;
    vector<uchar> status;
    vector<float> err;

    static Mat gray_prev;
    // static Mat stabframe_prev;
    Mat gray_curr;

    Mat stabframe(whole(ROI));
    

    if (stabframe.channels() == 3)
        cvtColor(stabframe, gray_curr, CV_BGR2GRAY);  // compute grayscale image
    else
        gray_curr = stabframe;

    
    if (!gray_prev.empty() && gray_prev.size() == gray_curr.size() && !features_prev.empty()) {
        // Feature matching using (sparse) optical flow of points using Lucas-Kanade-Method
        // void calcOpticalFlowPyrLK(InputArray prevImg, InputArray nextImg, InputArray prevPts, InputOutputArray nextPts, OutputArray status, OutputArray err,
        // Size winSize=Size(21,21), int maxLevel=3, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), int flags=0, double minEigThreshold=1e-4 )
        calcOpticalFlowPyrLK(gray_prev, gray_curr, features_prev, features_curr, status, err, Size(21,21), 3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), 0, 5e-4);
        clearBadFeatures(features_prev, features_curr, status, err);  // clear features with status==false

        try {
            if (!features_curr.empty()){
                stabilize(method, features_prev, features_curr, whole);  // stabilize frame using specified method and point pairs
            }
        } catch(cv::Exception) {}

    }

    // store for next loop, possible because of static variable
    gray_curr.copyTo(gray_prev);
    // stabframe.copyTo(stabframe_prev);
    //drawArrows(stabframe, features_curr, features_prev);
    vector<KeyPoint> keypoints = findFeatures(gray_curr, "PyramidHARRIS", features_prev);  // find new "prev" features in current image with specified featureDetector and store


    // drawPoints(stabframe, features_prev);
    drawKeypoints(whole, keypoints, whole, Scalar(255, 0, 255));
    if (!features_curr.empty()) drawPoints(whole, features_curr, Scalar(0, 255, 255));

    return whole;

}

void stabilize(string method, const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {

    /*
    if(method == "VIDEOSTAB"){
        stabilize_Videostab(feat_prev, feat_curr, whole);        
    }*/

    if(method == "PYR_LK_OPTICALFLOW"){
        stabilize_OpticalFlow(feat_prev, feat_curr, whole);
    }
    /*
    if (method == "RIGID_TRANSFORM"){
        stabilize_RigidTransform(feat_prev, feat_curr, whole);
    }

    if(method == "FIND_HOMOGRAPHY"){
        stabilize_Homography(feat_prev, feat_curr, whole);
    }*/

    else cout << "Chosen stationizing method no. " << method << " not associated with an implementation." << endl;
   
}


void stabilize_OpticalFlow(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {
    Point shift = getOpticalFlow(feat_prev, feat_curr);

    Mat trans_mat = (Mat_<float>(2, 3) << 1, 0, -shift.x, 0, 1, -shift.y);  // Creates a translation matrix with the mean optical flow
    warpAffine(whole, whole, trans_mat, whole.size());  // translates the whole frame with the matrix
}
/*
void stabilize_RigidTransform(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {
    static Mat prevMotions = Mat::eye(3, 3, CV_64F);

    if (RESET) prevMotions = Mat::eye(3, 3, CV_64F);

    Mat M = estimateRigidTransform(feat_prev, feat_curr, false);

    M.resize(3, 0);
    M.at<double>(2, 2) = 1;
    M.at<double>(0, 0) = M.at<double>(1, 1) = 1;
    M.at<double>(1, 0) = M.at<double>(0, 1) = 0;

    prevMotions = M * prevMotions;

    warpPerspective(whole, whole, prevMotions, whole.size(), WARP_INVERSE_MAP);
}

void stabilize_Homography(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {
    static Mat prevMotions = Mat::eye(3, 3, CV_64F);

    if (RESET) prevMotions = Mat::eye(3, 3, CV_64F);

    Mat M = findHomography(feat_prev, feat_curr, CV_RANSAC, 1);
    // Mat M = findHomography(feat_prev, feat_curr, CV_LMEDS);

    M.at<double>(2, 2) = 1;
    M.at<double>(2, 1) = M.at<double>(2, 0) = 0;
    M.at<double>(0, 0) = M.at<double>(1, 1) = 1;
    M.at<double>(1, 0) = M.at<double>(0, 1) = 0;

    double a = M.at<double>(0, 2);
    if (a*a > 100) M.at<double>(0, 2) = 0;
    a = M.at<double>(1, 2);
    if (a*a > 100) M.at<double>(1, 2) = 0;

    prevMotions = M * prevMotions;

    warpPerspective(whole, whole, prevMotions, whole.size(), WARP_INVERSE_MAP);
}



void stabilize_Videostab(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {
    // static Ptr<OnePassStabilizer> onePassStabilizer = new OnePassStabilizer();

    static Mat prevMotions = Mat::eye(3, 3, CV_32F);

    //if (RESET) {
    //    prevMotions = Mat::eye(3, 3, CV_32F);
    //} else {
        Mat M = cv::videostab::estimateGlobalMotionRobust(feat_prev, feat_curr, cv::videostab::TRANSLATION, cv::videostab::RansacParams(RANSAC_SIZE, RANSAC_THRESH*1./100, RANSAC_EPS*1./100, RANSAC_PROB*1./100));
        // Mat M = cv::videostab::estimateGlobalMotionLeastSquares(feat_prev, feat_curr, cv::videostab::TRANSLATION, 0);

        float a = M.at<float>(0, 2);
        if (a*a > 100) M.at<float>(0, 2) = 0;
        a = M.at<float>(1, 2);
        if (a*a > 100) M.at<float>(1, 2) = 0;

        prevMotions = M * prevMotions;
    //}
    warpPerspective(whole, whole, prevMotions, whole.size(), WARP_INVERSE_MAP);

    // k++;
}*/


void clearBadFeatures(vector<Point2f>& feat_prev, vector<Point2f>& feat_curr, vector<uchar>& stat, vector<float>& err) {
    // only applicable if stat and feat_curr is defined (through calcOpticalFlowPyrLK)
    assert(feat_prev.size() == feat_curr.size() && feat_prev.size() == stat.size());  // and the two features sets are the same size

    auto iter_1 = feat_prev.begin();
    auto iter_2 = feat_curr.begin();
    auto found = stat.begin();
    auto it_err = err.begin();

    while (iter_1 < feat_prev.end()) {
        if (!(static_cast<bool>(*found))) {
            // if status==false the feature was not found and feat_curr does not hold a proper point
            iter_1 = feat_prev.erase(iter_1);
            iter_2 = feat_curr.erase(iter_2);
            found = stat.erase(found);
            it_err = err.erase(it_err);
            continue;
        }
        if ( norm(*iter_1 - *iter_2) > 120 || *it_err > 1) {
            // remove features with an error greater than 1.0 because they are likely to be bad
            // and remove matched points with a distance greater that 120 for the same reason
            iter_1 = feat_prev.erase(iter_1);
            iter_2 = feat_curr.erase(iter_2);
            found = stat.erase(found);
            it_err = err.erase(it_err);
            continue;
        }
        ++iter_1;
        ++iter_2;
        ++found;
        ++it_err;
    }
}

vector<KeyPoint> findFeatures(Mat& img, string method, vector<Point2f>& feat){  // find good tracking Features in frame 'img' (grayscale)
    vector<KeyPoint> keypoints;

    Ptr<FeatureDetector> detector = FeatureDetector::create(method);
    /*Ptr<FeatureDetector> FeatureDetector::create(const string& detectorType)
    Parameters: 

    detectorType – Feature detector type.

    The following detector types are supported:
    
        "FAST" – FastFeatureDetector
        "STAR" – StarFeatureDetector
        "SIFT" – SIFT (nonfree module)
        "SURF" – SURF (nonfree module)
        "ORB" – ORB
        "BRISK" – BRISK
        "MSER" – MSER
        "GFTT" – GoodFeaturesToTrackDetector
        "HARRIS" – GoodFeaturesToTrackDetector with Harris detector enabled
        "Dense" – DenseFeatureDetector
        "SimpleBlob" – SimpleBlobDetector
    
    Also a combined format is supported: feature detector adapter name ( "Grid" – GridAdaptedFeatureDetector, "Pyramid" – 
    PyramidAdaptedFeatureDetector ) + feature detectorname (see above), for example: "GridFAST", "PyramidSTAR" . */

    detector->detect(img, keypoints);

    KeyPoint::convert(keypoints, feat);  // store as Point2f
    return keypoints;
}

void drawArrows(Mat& drawframe, const vector<Point2f>& feat_1, const vector<Point2f>& feat_2) {
    assert(feat_1.size() == feat_2.size());  // zeichne Pfeile zwischen den zugeordneten Paaren von Punkten

    for (unsigned int i = 0; i < feat_1.size(); i++) {
        arrowedLine(drawframe, feat_1[i], feat_2[i], Scalar(255, 0, 0), 1, CV_AA, 0, 0.3);
    }
}

void drawPoints(Mat& drawframe, const vector<Point2f>& feat, Scalar s) {
    if (feat.empty()) cout << "Vector of Point2f \"feat\" empty." << endl;
    for (const Point2f& x : feat) {
        circle(drawframe, x, 2, s, -1, CV_AA, 0);
    }
}

Point2f getOpticalFlow(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr) {  // calculate mean differences of the feature points
    assert(feat_prev.size() == feat_curr.size());

    // Averaging approach
    float x_diff_mean(0), y_diff_mean(0);
    int n(0);


    for (unsigned int i = 0; i < feat_prev.size(); i++) {
        // Averaging approach
        n++;
        x_diff_mean += feat_curr[i].x - feat_prev[i].x;
        y_diff_mean += feat_curr[i].y - feat_prev[i].y;
    }

    // Averaging approach
    return Point2f(x_diff_mean/n, y_diff_mean/n);
}













