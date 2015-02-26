/*  Author: Philipp Zander
    at TU Dortmund, Physik, Experimentelle Physik 5
    year: 2014
*/

#include <TH2D.h>
#include <TCanvas.h>
#include <opencv2/highgui/highgui.hpp>  // GUI windows and trackbars
#include <opencv2/video/video.hpp>  // camera/video classes, imports background_segm
#include <opencv2/calib3d/calib3d.hpp>  // for FindHomography
#include <opencv2/features2d/features2d.hpp>  // for FeatureDetection
#include <opencv2/videostab/stabilizer.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>  // used for putting text on the frames
#include <fstream>  // reading/writing calibration files
#include <math.h>

//using namespace std;
using namespace cv;
using std::cout; using std::cin; using std::endl; using std::stringstream; using std::vector; using std::ifstream; using std::ofstream;

// global variables ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
double cam_height, cam_width;
int bw_tresh, canny_tresh, center_tresh, min_dist, min_radius, max_radius, accumulator_res;
int roi_lower, roi_upper, roi_left, roi_right, voting_binsize, ransac_size;
int ransac_thresh, ransac_eps, ransac_prob, MOG_learningrate;
int h_min, h_max, v_min, v_max, s_min, s_max;
int b_min, b_max, g_min, g_max, r_min, r_max;
int wait_ms = 10;
float mean_fiber_distances;
string outputName, inputName;

enum StabilizationMethod {PYR_LK_OPTICALFLOW, RIGID_TRANSFORM, FIND_HOMOGRAPHY, VIDEOSTAB, PHASE_CORRELATION};

// function declarations – function used by main –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void usage() {  // display the programs usage
    cout << "\033[00;30m This Program was written at the\033[1m Chair for Experimental Physics V at the TU Dortmund\033[0m" << endl;
    // cout << "If not provided an input file it will compute the stream from the main camera attached to your system at default. ";
    cout << "Press ESC or q to exit the program" << endl << "Usage: " << endl;
    // cout << "\t\t >> Run from /videos/**/ <<" << endl;
    cout << "\t -input    \t[-i] \t **Not implemented** Input video file" << endl;
    cout << "\t -calibrate\t[-c] \t Calibrate the main parameters of the program. " << endl;
    cout << "\t\t\t\t If used there should be a '/calib' dir present." << endl;
    cout << "\t -help     \t[-h] \t Display this message" << endl;
}
void stabilizeFrame(Mat&);
void processFrame(Mat&);
void correctValues(int, void*);  // setting correct values for some parameters
void setDefaults();  // set default values for all parameters
void createCalibWindow();  // create the window where the calibration trackbars should be displayed (calibrate==true)
void readCalib();  // read the calibration parameters from file
void setDefaults();
void writeCalib();  // write the calibration parameters to another file

void OnClick(int event, int x, int y, int, void* frame) {
     if  ( event == EVENT_LBUTTONDOWN ) {
          cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
     }
}

// main ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
int main(int argc, char *argv[]) {
    bool calibrate = false, input = true;  // values that might later be changed through passing arguments
    char key;
    string videoName;

    videoName  = "../videos/video1/frame_%03d.jpg";  // hardcoded paths to video and calibration files are going to be
    inputName  = "../calib/calibration00.txt";  // replaced with the required input of paths or names (i.e. CV_CAP_ANY)

    cout <<"\033[00;30m";

    if (argc > 2) {
        cout << argc << " is not a valid number of arguments. EXIT" << endl;
        usage();
        exit(EXIT_FAILURE);
    } else {
        if (argc == 2) {
            if (string(argv[1]) == string("-calibrate") || string(argv[1]) == string("-c")) {
                calibrate = true;
            } else {
                if (string(argv[1]) == string("-help") || string(argv[1]) == string("-h")) {
                    usage();
                    exit(EXIT_SUCCESS);
                } else {
                    cout << argv[1] << " is not a valid argument. EXIT" << endl;
                    usage();
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    // VideoCapture cam(CV_CAP_ANY);  // Capture using any camera connected
    VideoCapture cam(videoName);  // Load video file or a sequence of images
    Mat whole;  // container for the 'whole' video frame
    
    // namedWindow("Video Frame", WINDOW_NORMAL|CV_WINDOW_KEEPRATIO);

    if (!cam.isOpened()) {
        cout << "capture is not open. BREAK" << endl;
        exit(EXIT_FAILURE);
    }

    cam >> whole; // Create first frame from capture/video; without this the HEIGHT/WIDTH etc. will not be defined

    // get the format of the video/image stream
    cam_height = cam.get(CV_CAP_PROP_FRAME_HEIGHT);
    cam_width  = cam.get(CV_CAP_PROP_FRAME_WIDTH);

    cout << "Video format is h: " << cam_height << ", w: " << cam_width << endl;

    if (input) readCalib();
    else setDefaults();

    h_min = 30;
    v_min = s_min = 65;
    h_max = v_max = s_max = 150;

    b_min = g_max = 120;
    g_min = r_min = 80;
    b_max = r_max = 160;

    if (calibrate) createCalibWindow();

    cout << "Loop starts..." << endl;
    while (key != 27 && key != 'q') {  // Create loop for streaming
        // end loop when 'ESC' or 'q' is pressed

        if (whole.empty()) {
            cam.open(videoName);
            cam >> whole;
            continue;
        }

        GaussianBlur(whole, whole, Size(7, 7), 4);  // blurring reduces noise
        // medianBlur(binary, median, 5);

        stabilizeFrame(whole);  // stabile the current frame
        processFrame(whole);  // process the stabilized frame

        double fps = cam.get(CV_CAP_PROP_FPS);
        double pos = cam.get(CV_CAP_PROP_POS_FRAMES);

        Rect white(0,0,100,60);
        rectangle(whole, white, Scalar::all(0), CV_FILLED, CV_AA, 0);

        stringstream s;
        s << "fps: " << fps;
        putText(whole, s.str(), Point(10,25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1);
        s.str("");
        s << "no: " << pos;
        putText(whole, s.str(), Point(10,50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1);

        imshow("Video Frame", whole);  // show the results in windows
        createTrackbar("waitKey duration:", "Video Frame", &wait_ms, 1000, nullptr);

        key = waitKey(wait_ms*10);  // Capture Keyboard stroke

        cam >> whole;  // Get next frame; at the end of the loop because the first frame was retrieved before the loop started
    }
    cout << "Loop ended. Releasing capture and destroying windows..." << endl;

    cam.release();
    startWindowThread();  // Must be called to destroy the windows properly
    destroyAllWindows();  // otherwise the windows will remain openend.

    while (calibrate && key != 'n' && key != 'N') {
        cout << "Do you want to store the new calibration? (y/n) ";
        cin >> key;
        if (key == 'y' || key == 'Y') {
            cout << "Type the path of the new calibration file: (type '-d' for default file '../calib/calibration00.txt')" << endl;
            cin >> outputName;
            if (outputName == string("-d"))
                outputName = "../calib/calibration00.txt";
            writeCalib();
            key = 'n';
        }
    }

    exit(EXIT_SUCCESS);
}
// function declarations – functions used for actual processing ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void findFeatures(Mat&, string method, vector<Point2f>&);
void clearFeaturesNotFound(vector<Point2f>&, vector<Point2f>&, vector<uchar>&, vector<float>&);
// void eraseLong(float);
void stabilize(int method, vector<Point2f>&, vector<Point2f>&, Mat&);
void drawCircles(Mat&, vector<Vec3f>&, Scalar = Scalar(0, 0, 255));
void drawContours(Mat&);
void drawPoints(Mat&, vector<Point2f>&);
void drawArrows(Mat&, vector<Point2f>&, vector<Point2f>&);
Point2f getOpticalFlow(vector<Point2f>&, vector<Point2f>&);
Point2f getDenseOpticalFlowVoting(Mat&);
void showOpticalFlow(Mat&, vector<Point2f>&, vector<Point2f>&);
void convertDenseOpticalFlow(Mat&, Mat&);
void deleteOutliers(vector<Vec3f>&);  // 
void updateHistogram(vector<Vec3f>&);
void calcDifferences(vector<Vec3f>&, vector<float>&); // calculates the differences of the centers of circles
void sortCircles(vector<Vec3f>& circles) {  // sort circles by their x-coordinate in ascending order
    std::sort(circles.begin(), circles.end(), [](Vec3f a, Vec3f b){ return a[0] < b[0]; });}
void getRelevantFlow(Mat&, vector<Point2f>&, vector<Point2f>&);

// actual doing something with the image –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void processFrame(Mat& whole) {
    Mat frame, drawframe, gray, binary, median, gauss, edges;

    //––– define region of interest (ROI) with parameters set by trackbars ––––––––––––––––––––––––––––––––––––––––––
    Rect ROI(Point(cam_width/2-roi_left, cam_height/2+roi_upper), Point(cam_width/2+roi_right, cam_height/2-roi_lower));
    drawframe = whole(ROI);  // extract ROI 
    drawframe.copyTo(frame);  // split in two frames; one for processing and one for drawing and showing 
    imshow("Region of interest", frame);

    cvtColor(frame, gray, CV_BGR2GRAY);  // compute grayscale image
    imshow("Grayscale image", gray);

    rectangle(whole, ROI, Scalar(0, 255, 0), 1, CV_AA, 0);  // draw processed region of interest on whole image

    threshold(gray, binary, bw_tresh, 255, THRESH_BINARY_INV);  // compute b/w img
    imshow("Binary image", binary);

    //––– Use background substraction for detection of large errors –––––––––––––––––––––––––––––––––––––––––––––––––
    Mat fgMOG, fgMOG2, fgGMG, fgAll;
    static BackgroundSubtractorMOG  MOG;  // MOG approach 
    static BackgroundSubtractorMOG2 MOG2(10, 16/*:default values*/, false/*=doShadowDetection*/);  // MOG2 approach
    static BackgroundSubtractorGMG  GMG;  // GMG approach 

    // Background substraction of the live image using MOG and MOG2
    MOG(frame, fgMOG, MOG_learningrate*1./100);
    MOG2(frame, fgMOG2, MOG_learningrate*1./100);
    GMG(frame, fgGMG, MOG_learningrate*1./100);
    multiply(fgMOG, fgMOG2, fgAll);
    multiply(fgGMG, fgAll, fgAll);

    Rect white(0,0,200,30);
    rectangle(fgMOG, white, Scalar::all(0), CV_FILLED, CV_AA, 0);
    rectangle(fgMOG2, white, Scalar::all(0), CV_FILLED, CV_AA, 0);
    rectangle(fgGMG, white, Scalar::all(0), CV_FILLED, CV_AA, 0);
    rectangle(fgAll, white, Scalar::all(0), CV_FILLED, CV_AA, 0);
    
    stringstream s;
    s << "norm: " << norm(fgMOG);
    putText(fgMOG, s.str(), Point(10,25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1);
    s.str("");
    s << "norm: " << norm(fgMOG2);
    putText(fgMOG2, s.str(), Point(10,25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1);
    s.str("");
    s << "norm: " << norm(fgGMG);
    putText(fgGMG, s.str(), Point(10,25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1);
    s.str("");
    s << "norm: " << norm(fgAll);
    putText(fgAll, s.str(), Point(10,25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1);
    s.str("");

    imshow("Foreground Mask MOG", fgMOG);
    imshow("Foreground Mask MOG2", fgMOG2);
    imshow("Foreground Mask GMG", fgGMG);
    imshow("FG multiplication MOG*MOG2", fgAll);

    //––– Detect edges in the blurred binary image using Canny algorithm ––––––––––––––––––––––––––––––––––––––––––––
    Canny(binary, edges, canny_tresh, 2*canny_tresh, 5);
    imshow("Edges", edges);

    //––– Detect circles in the binary image using the Hough transformation –––––––––––––––––––––––––––––––––––––––––
    vector<Vec3f> circles;  // Vektor für die gefundenen Kreise
    HoughCircles(binary, circles, CV_HOUGH_GRADIENT, accumulator_res, min_dist, 2*canny_tresh, center_tresh, min_radius, max_radius);
    sortCircles(circles);  // sorting according to x-coordinate
    deleteOutliers(circles);

    drawCircles(drawframe, circles);
    // cout << " [";
    // for(double& d: differences) cout << d << ",";
    // cout << "] with minimum " << *std::min_element(differences.begin(), differences.end()) << endl;

    //––– Somehow update the Histogram containing the circle center differences
    updateHistogram(circles);

    // findContours(binary, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    // drawContours(drawframe);

    // if (features_curr.size() == features_found.size()) showOpticalFlow();

    Mat whole_hsv_thresh, whole_bgr_thresh;
    cvtColor(whole,whole_hsv_thresh,CV_BGR2HSV);
    inRange(whole_hsv_thresh, Scalar(h_min, s_min, v_min), Scalar(h_max,s_max,v_max), whole_hsv_thresh);
    inRange(whole, Scalar(b_min, g_min, r_min), Scalar(b_max,g_max,r_max), whole_bgr_thresh);

    imshow("BGR thresholded image", whole_bgr_thresh);
    imshow("HSV thresholded image", whole_hsv_thresh);

    // pointing at the primary position of 'the blob' for testing purposes
    int x = 515, y = 205;
    line(whole, Point(x, y-30), Point(x, y+30), Scalar(255, 0, 0), 1, CV_AA);
    line(whole, Point(x-20, y), Point(x+20, y), Scalar(255, 0, 0), 1, CV_AA);
}
void stabilizeFrame(Mat& whole) {
    static vector<Point2f> features_prev;
    vector<Point2f> features_curr;
    vector<uchar> status;
    vector<float> err;

    static Mat gray_prev;
    static Mat stabframe_prev;
    Mat gray_curr;

    Rect ROI(Point(cam_width/2-roi_left, cam_height/2+roi_upper), Point(cam_width/2+roi_right, cam_height/2-roi_lower));
    Mat stabframe = whole(ROI).clone();

    cvtColor(stabframe, gray_curr, CV_BGR2GRAY);  // compute grayscale image

    if (gray_prev.size() == gray_curr.size()) {
        //––– Sparse Optical Flow –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        // Feature matching using (sparse) optical flow of points using Lucas-Kanade-Method
        calcOpticalFlowPyrLK(gray_prev, gray_curr, features_prev, features_curr, status, err);
        clearFeaturesNotFound(features_prev, features_curr, status, err);  // clear features with status==false

        stabilize(RIGID_TRANSFORM, features_prev, features_curr, whole);

        //––– Dense Optical Flow ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        // Dense optical flow calculation using Farneback's method or the TVL1 method, no feature points involved
        static Mat flow(gray_prev.size(), CV_32FC2);
        // void calcOpticalFlowFarneback(InputArray prev, InputArray next, InputOutputArray flow, double pyr_scale, int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags)
        calcOpticalFlowFarneback(gray_prev, gray_curr, flow, 0.5, 5, 15, 3, 5, 1.1, OPTFLOW_USE_INITIAL_FLOW);
        // static Ptr<DenseOpticalFlow> optflowcalculator = createOptFlow_DualTVL1();
        // optflowcalculator->calc(gray_prev, gray_curr, flow);
        // calcOpticalFlowSF(stabframe_prev, stabframe, flow, 3, 2, 4);

        // medianBlur(flow, flow, 5);

        Mat bgr(flow.size(), CV_32F);
        convertDenseOpticalFlow(flow, bgr);
        imshow("Dense optical flow", bgr);

        // getRelevantFlow(flow, features_prev, features_curr);
        // drawArrows(stabframe, features_curr, features_prev);
        // stabilize(VIDEOSTAB, features_prev, features_curr, whole);
    }

    // store for next loop
    gray_curr.copyTo(gray_prev);
    stabframe.copyTo(stabframe_prev);
    findFeatures(gray_curr, "PyramidGFTT", features_prev);  // find new "prev" features in current image and store

    drawPoints(stabframe, features_prev);
    imshow("Stabilizing frame", stabframe);
}
void findFeatures(Mat& img, string method, vector<Point2f>& feat){  // find good tracking Features in frame 'img' (grayscale)
    vector<KeyPoint> keypoints;

    Ptr<FeatureDetector> detector = FeatureDetector::create(method);
    detector->detect(img, keypoints);

    KeyPoint::convert(keypoints, feat);  // store as Point2f

    // if (draw) drawKeypoints(drawframe, keypoints, drawframe, Scalar(0, 255, 255));
}
void clearFeaturesNotFound(vector<Point2f>& feat_prev, vector<Point2f>& feat_curr, vector<uchar>& stat, vector<float>& err) {  // only applicable if stat and feat_curr is defined (through calcOpticalFlowPyrLK)
    assert (feat_prev.size() == feat_curr.size() && feat_prev.size() == stat.size());  // and the two features sets are the same size

    vector<Point2f>::iterator iter_1 = feat_prev.begin();
    vector<Point2f>::iterator iter_2 = feat_curr.begin();
    vector<uchar>::iterator found = stat.begin();
    vector<float>::iterator it_err = err.begin();

    while (iter_1 < feat_prev.end()) {
        if (!(bool)(*found)) {
            iter_1 = feat_prev.erase(iter_1);
            iter_2 = feat_curr.erase(iter_2);
            found = stat.erase(found);
            it_err = err.erase(it_err);
        } else {
            ++iter_1;
            ++iter_2;
            ++found;
            ++it_err;
        }
    }
}
void getRelevantFlow(Mat& flow, vector<Point2f>& prev, vector<Point2f>& curr) {
    assert (!prev.empty() && !flow.empty());

    Mat xy[2]; //X,Y
    split(flow, xy);
    curr.clear();

    for (vector<Point2f>::iterator i = prev.begin(); i != prev.end();) {
        float x = (*i).x; float y = (*i).y;

        if (x > flow.rows || y > flow.cols) {
            i = prev.erase(i);
        } else {
            float new_x = x + xy[0].at<float>(x,y);
            float new_y = y + xy[0].at<float>(x,y);

            curr.push_back(Point2f(new_x, new_y));
            ++i;
        }
    }
}
// void eraseLong(float threshold) {
//     assert (features_prev.size() == features_curr.size() && status.size() == features_curr.size());

//     vector<Point2f>::iterator iter_1 = features_prev.begin();
//     vector<Point2f>::iterator iter_2 = features_curr.begin();
//     vector<uchar>::iterator found = status.begin();
//     vector<float>::iterator it_err = err.begin();

//     //vector<Point2f>::iterator end = features_prev.end();
//     while (iter_1 < features_prev.end()) {

//         float x_diff = (*iter_2).x - (*iter_1).x;
//         float y_diff = (*iter_2).y - (*iter_1).y;
//         float c = pow((pow(x_diff, 2) + pow(y_diff, 2)), 0.5);

//         if (threshold < c) {
//             iter_1 = features_prev.erase(iter_1);
//             iter_2 = features_curr.erase(iter_2);
//             found = status.erase(found);
//             it_err = err.erase(it_err);
//         } else {
//             ++iter_1;
//             ++iter_2;
//             ++found;
//             ++it_err;
//         }
//     }
// }
void stabilize_OpticalFlow(vector<Point2f>& feat_prev, vector<Point2f>& feat_curr, Mat& whole) {
    Point shift = getOpticalFlow(feat_prev, feat_curr);

    Mat trans_mat = (Mat_<float>(2,3) << 1, 0, -shift.x, 0, 1, -shift.y);  // Creates a translation matrix with the mean optical flow
    warpAffine(whole, whole, trans_mat, whole.size());  // translates the whole frame with the matrix
}
void stabilize_RigidTransform(vector<Point2f>& feat_prev, vector<Point2f>& feat_curr, Mat& whole) {
    static Mat prevMotions = Mat::eye(3, 3, CV_64F);

    Mat M = estimateRigidTransform(feat_prev, feat_curr, false);

    M.resize(3, 0);
    M.at<double>(2, 2) = 1;
    M.at<double>(0,0) = M.at<double>(1,1) = 1;
    M.at<double>(1,0) = M.at<double>(0,1) = 0;

    prevMotions = M * prevMotions;

    // cout << "[" << M.at<double>(0,2) << ", "<< M.at<double>(1,2) << "]" << endl << endl;

    warpPerspective(whole, whole, prevMotions, whole.size(), WARP_INVERSE_MAP);
}
void stabilize_Homography(vector<Point2f>& feat_prev, vector<Point2f>& feat_curr, Mat& whole) {
    Mat M = findHomography(feat_prev, feat_curr, CV_RANSAC);
    warpPerspective(whole, whole, M, whole.size(), WARP_INVERSE_MAP);
}
// void stabilize_PhaseCorrelation(Mat& whole) {
//     Point2d shift = phaseCorrelate(gray_first, gray);
//     cout << shift << endl;

//     Mat trans_mat = (Mat_<float>(2,3) << 1, 0, shift.x, 0, 1, shift.y);  // Creates a translation matrix with the mean optical flow
//     warpAffine(whole, whole, trans_mat, whole.size(), WARP_INVERSE_MAP);  // translates the whole frame with the matrix
// }
void stabilize_Videostab(vector<Point2f>& feat_prev, vector<Point2f>& feat_curr, Mat& whole) {
    using namespace cv::videostab;
    //static Ptr<OnePassStabilizer> onePassStabilizer = new OnePassStabilizer();

    static Mat prevMotions = Mat::eye(3,3,CV_32FC1);
    // static vector<Mat> motions;  // here the motion prev->curr must be estimated
    // static int k = 1;

    Mat M = estimateGlobalMotionRobust(feat_prev, feat_curr, TRANSLATION, RansacParams(ransac_size, ransac_thresh*1./100, ransac_eps*1./100, ransac_prob*1./100));
    prevMotions = M * prevMotions;

    warpPerspective(whole, whole, prevMotions, whole.size(), WARP_INVERSE_MAP);

    // k++;
}
void stabilize(int method, vector<Point2f>& feat_prev, vector<Point2f>& feat_curr, Mat& whole) {
    switch (method) {
        case PYR_LK_OPTICALFLOW: stabilize_OpticalFlow(feat_prev, feat_curr, whole);    break;
        case RIGID_TRANSFORM:    stabilize_RigidTransform(feat_prev, feat_curr, whole); break;
        case FIND_HOMOGRAPHY:    stabilize_Homography(feat_prev, feat_curr, whole);     break;
        case VIDEOSTAB:          stabilize_Videostab(feat_prev, feat_curr, whole);      break;
        // case PHASE_CORRELATION:  stabilize_PhaseCorrelation(whole);                     break;
        default: cout << "Chosen stationizing method no. " << method << " not associated with an implementation." << endl;
    }
}
Point2f getOpticalFlow(vector<Point2f>& feat_prev, vector<Point2f>& feat_curr) {  // calculate mean differences of the feature points
    assert (feat_prev.size() == feat_curr.size());

    // Averaging approach
    float x_diff_mean, y_diff_mean;
    int n = 0;
    x_diff_mean = y_diff_mean = 0;

    // Voting approach
    // Size fsize = frame.size();
    // TH2D votingHist("votingHist", "", fsize.width*2./voting_binsize, -fsize.width, fsize.width,
                                      // fsize.height*2./voting_binsize, -fsize.height, fsize.height);

    for(unsigned int i = 0; i < feat_prev.size(); i++) {
        // If Pyramidal Lucas Kanade didn't really find the feature, skip it 
        // if (status[i] == false)  // removed due to the function 'clearFeaturesNotFound'
        //     continue;

        // Averaging approach
        n++;
        x_diff_mean += feat_curr[i].x - feat_prev[i].x;
        y_diff_mean += feat_curr[i].y - feat_prev[i].y;

        // Voting approach
        // float x_diff = feat_curr[i].x - feat_prev[i].x;
        // float y_diff = feat_curr[i].y - feat_prev[i].y;
        // votingHist.Fill(x_diff,y_diff);
    }

    // Averaging approach
    return Point2f(x_diff_mean/n, y_diff_mean/n);

    // int locmax, locmay, locmaz;
    // votingHist.GetMaximumBin(locmax, locmay, locmaz);
    // double max = votingHist.GetXaxis()->GetBinCenter(locmax);
    // double may = votingHist.GetYaxis()->GetBinCenter(locmay);
    // Point2f maxLoc(max, may);
    // cout << "Maximum Votes at " << maxLoc  << " contains " << votingHist.GetMaximum()*100./feat_prev.size() << "% of the found keypoints" << endl;
    // return maxLoc;
}
Point2f getDenseOpticalFlowVoting(Mat& flow) {
    double voting_binsize = 0.1;
    // Voting approach
    Size fsize(cam_height, cam_width);
    TH2D votingHist("votingHist", "", fsize.width*2./voting_binsize, -fsize.width, fsize.width,
                                      fsize.height*2./voting_binsize, -fsize.height, fsize.height);

    Mat xy[2]; //X,Y
    split(flow, xy);

    for(int i = 0; i < flow.size().height; i++) {
        for (int j = 0; j < flow.size().width; j++) {
            double x_diff = xy[0].at<float>(i,j);
            double y_diff = xy[1].at<float>(i,j);
            if (x_diff == 0 && y_diff == 0) continue;
            votingHist.Fill(x_diff, y_diff);
        }
    }

    int locmax, locmay, locmaz;
    votingHist.GetMaximumBin(locmax, locmay, locmaz);
    double max = votingHist.GetXaxis()->GetBinCenter(locmax);
    double may = votingHist.GetYaxis()->GetBinCenter(locmay);
    Point2f maxLoc(max, may);
    // cout << "Maximum Votes at " << maxLoc  << " contains " << votingHist.GetMaximum()*100./feat_prev.size() << "% of the found keypoints" << endl;
    cout << "Maximum Votes at " << maxLoc  << " contains " << votingHist.GetMaximum()*100./flow.size().area() << "% of the flow points" << endl;
    return maxLoc;
}
template<typename T, int size>
vector<T> extractValues(vector<Vec<T, size>> vec, int i) {
    vector<T> returnvec;
    for (Vec<T, size>& x: vec) returnvec.push_back(x[i]);
    return returnvec;
}
// double calculateMean() {
//     double min = 0.0;

// }
template<typename T>
void print(vector<T>& vec) {
    for(T& x: vec) cout << x << " ";
}
void updateHistogram(vector<Vec3f>& circles) {
    if (circles.empty()) return;
    static TH1D centerDistances("centerDistances", "Differences of Fiber Threads", 100, 30, 90);

    vector<float> circle_differences;
    calcDifferences(circles, circle_differences);
    cout << "Center Differences: [";
    print<float>(circle_differences);
    cout << "]" << endl;

    mean_fiber_distances = centerDistances.GetMean();
}
void calcDifferences(vector<Vec3f>& circles, vector<float>& out) {
// calculate the differences of the circle centers assuming the circles are ordered ascending according to their x-coordinate
    out.clear();
    if (circles.empty()) return;

    Vec3f last = circles[0];

    for(uint i=1; i < circles.size(); i++) {
        Vec3f c = circles[i];

        float diff = pow(pow(c[0] - last[0], 2) + pow(c[1] - last[1], 2), 0.5);
        out.push_back(diff);
        last = c;
    }
}
void deleteOutliers(vector<Vec3f>& circles) {
// approach to delete the outlier circles, that are circles not corresponding to a fiber thread, assuming that most
// of the circles are corresponding to a thread, using 'least squares' for just the y-coordinates (sorted in x)
    if (circles.empty()) return;
    vector<float> y_coordinates = extractValues<float, 3>(circles, 1);

    int max_x = cvRound(circles.back()[0]);
    int max_y = cvRound(*max_element(y_coordinates.begin(), y_coordinates.end()));
    Size matsize(max_x+2*min_radius, max_y+2*min_radius);
    static Mat drawframe(matsize, CV_8UC3);

    if (matsize.height > drawframe.size().height) drawframe.resize(matsize.height, 0);
    if (matsize.width > drawframe.size().width) {
        drawframe = drawframe.t();
        drawframe.resize(matsize.width, 0);
        drawframe = drawframe.t();
    }
    drawframe *= 0;

    float y_coord_mean = 0;
    for(float& y: y_coordinates) y_coord_mean += y;
    y_coord_mean /= y_coordinates.size();

    int threshold = 1./2*min_radius;

    line(drawframe, Point(0, y_coord_mean), Point(drawframe.cols, y_coord_mean), Scalar::all(128));
    line(drawframe, Point(0, y_coord_mean+threshold), Point(drawframe.cols, y_coord_mean+threshold), Scalar::all(128), 1, CV_AA);
    line(drawframe, Point(0, y_coord_mean-threshold), Point(drawframe.cols, y_coord_mean-threshold), Scalar::all(128), 1, CV_AA);

    // Point2f last(circles[0][0], circles[0][1]);
    // for(uint i = 1; i < circles.size(); i++) {
    //     Point2f c(circles[i][0], circles[i][1]);
    //     line(drawframe, last, c, Scalar::all(255));
    //     last = c;
    // }

    // vector<Vec4i> lines;
    // HoughLinesP(drawframe, lines, 1, CV_PI/4, 20, 0, 0 );

    // for(Vec4i& l: lines) line(drawframe, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(128), 3, CV_AA);

    // Results of the procedure, show rejected circles and inliners on the frame and remove outliers
    drawCircles(drawframe, circles);
    circles.erase(remove_if(circles.begin(), circles.end(), [y_coord_mean, threshold](Vec3f a){ return !(a[1] < (y_coord_mean + threshold) && a[1] > (y_coord_mean - threshold)); }), circles.end());
    drawCircles(drawframe, circles, Scalar(0, 255, 0));

    imshow("Circle outlier detection", drawframe);
}
// a bunch of drawing functions ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void drawCircles(Mat& drawframe, vector<Vec3f>& circles, Scalar color) {
    for (const Vec3f& circ: circles) {
        // speichert die ersten beiden Einträge von vector<Vec3f> circles in center und den dritten Wert in radius
        Point center(circ[0], circ[1]);
        int radius = circ[2];

        circle(drawframe, center, radius, color, 1, CV_AA, 0);  // draw circle
        circle(drawframe, center, 2, color, -1, CV_AA, 0);      // draw center
    }
}
void drawPoints(Mat& drawframe, vector<Point2f>& feat) {
    if (feat.empty()) cout << "Vector of Point2f \"feat\" empty." << endl;
    for (Point2f& x: feat) {
        circle(drawframe, x, 2, Scalar(0, 255, 255), -1, CV_AA, 0);
    }
}
void drawArrows(Mat& drawframe, vector<Point2f>& feat_1, vector<Point2f>& feat_2) {
    assert (feat_1.size() == feat_2.size());  // zeichne Pfeile zwischen den zugeordneten Paaren von Punkten

    for(unsigned int i = 0; i < feat_1.size(); i++) {
        arrowedLine(drawframe, feat_1[i], feat_2[i], Scalar(255,0,0), 1, CV_AA, 0, 0.3);
    }
}
void showOpticalFlow(Mat& drawframe, vector<Point2f>& feat_prev, vector<Point2f>& feat_curr) {
    Point shift = getOpticalFlow(feat_prev, feat_curr);
    Rect white(0,0,200,60);
    rectangle(drawframe, white, Scalar::all(0), CV_FILLED, CV_AA, 0);
    
    stringstream s;
    s.str("");
    s << "x_shift: " << shift.x;
    putText(drawframe, s.str(), Point(10,25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1);
    s.str("");
    s << "y_shift: " << shift.y;
    putText(drawframe, s.str(), Point(10,50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1);
}
void convertDenseOpticalFlow(Mat& flow, Mat& out) {
    Mat xy[2]; //X,Y
    split(flow, xy);
    
    //calculate angle and magnitude
    Mat magnitude, angle;
    cartToPolar(xy[0], xy[1], magnitude, angle, true);
    
    //translate magnitude to range [0;1]
    double mag_max;
    minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0/mag_max);
    
    //build hsv image
    Mat _hls[3], hsv;
    _hls[0] = angle;
    _hls[1] = Mat::ones(angle.size(), CV_32F);
    _hls[2] = magnitude;
    merge(_hls, 3, hsv);

    cvtColor(hsv, out, COLOR_HSV2BGR);
}
// calibration functions –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void correctValues(int, void*) {  // setting correct (>0) value for some parameters
    if (canny_tresh == 0)
        canny_tresh++;
    if (center_tresh == 0)
        center_tresh++;
    if (accumulator_res == 0)
        accumulator_res++;
    if (roi_lower < 10)
        roi_lower = 10;
    if (roi_upper < 10)
        roi_upper = 10;
    if (roi_left < 10)
        roi_left = 10;
    if (roi_right < 10)
        roi_right = 10;
    if (voting_binsize == 0)
        voting_binsize++;
    if (ransac_size == 0)
        ransac_size++;
    if (ransac_eps == 0)
        ransac_eps++;
    if (ransac_thresh == 0)
        ransac_thresh++;
    if (ransac_prob == 0)
        ransac_prob++;
    if (min_dist < 15)  // not necessarily, but a smaller value slows down the program
        min_dist = 15;

    // update the trackbar positions
    setTrackbarPos("Center treshhold:", "Calibration window", center_tresh);
    setTrackbarPos("Canny lower treshhold:", "Calibration window", canny_tresh);
    setTrackbarPos("Reciprocal accumulator resolution:", "Calibration window", accumulator_res);
    setTrackbarPos("Lower boundary of ROI:", "Calibration window", roi_upper);
    setTrackbarPos("Upper boundary of ROI:", "Calibration window", roi_lower);
    setTrackbarPos("Left boundary of ROI: ",  "Calibration window", roi_left);
    setTrackbarPos("Right boundary of ROI:", "Calibration window", roi_right);
    setTrackbarPos("Minimum distance between circles:", "Calibration window", min_dist);

    setTrackbarPos("Binsize of Voting Histogram:", "Calibration window", voting_binsize);

    setTrackbarPos("Subset size:", "RANSAC Parameters", ransac_size);
    setTrackbarPos("Error threshold (%):", "RANSAC Parameters", ransac_thresh);
    setTrackbarPos("Maximum outlier ratio (%):", "RANSAC Parameters", ransac_eps);
    setTrackbarPos("Success probability (%):", "RANSAC Parameters", ransac_prob);
}
// functions opening and updating windows ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void createCalibWindow() {  // create the window where the calibration trackbars should be displayed (calibrate==true)
    namedWindow("Calibration window" , WINDOW_NORMAL);
    namedWindow("RANSAC Parameters", WINDOW_NORMAL);
    namedWindow("HSV Thresholds", WINDOW_NORMAL);
    namedWindow("BGR Thresholds", WINDOW_NORMAL);

    createTrackbar("Lower boundary of ROI:", "Calibration window", &roi_upper, cam_height/2, correctValues);
    createTrackbar("Upper boundary of ROI:", "Calibration window", &roi_lower, cam_height/2, correctValues);
    createTrackbar("Left boundary of ROI: ", "Calibration window", &roi_left,  cam_width/2,  correctValues);
    createTrackbar("Right boundary of ROI:", "Calibration window", &roi_right, cam_width/2,  correctValues);

    createTrackbar("B/W treshhold:", "Calibration window", &bw_tresh, 255, nullptr);

    createTrackbar("Canny lower treshhold:", "Calibration window", &canny_tresh, 1000, correctValues);
    createTrackbar("Center treshhold:", "Calibration window", &center_tresh, 100, correctValues);
    createTrackbar("Reciprocal accumulator resolution:", "Calibration window", &accumulator_res, 5, correctValues);
    createTrackbar("Minimum distance between circles:", "Calibration window", &min_dist, 255, correctValues);
    createTrackbar("Minimum radius of circles:", "Calibration window", &min_radius, 100, nullptr);
    createTrackbar("Maximum radius of circles:", "Calibration window", &max_radius, 100, nullptr);

    createTrackbar("Binsize of Voting Histogram:", "Calibration window", &voting_binsize, 100, correctValues);
    createTrackbar("MOG1/2 Learning rate (%): ", "Calibration window", &MOG_learningrate, 99, nullptr);

    createTrackbar("Subset size:",     "RANSAC Parameters", &ransac_size, 10, correctValues);
    createTrackbar("Error threshold (%):", "RANSAC Parameters", &ransac_thresh, 99, correctValues);
    createTrackbar("Maximum outlier ratio (%):", "RANSAC Parameters", &ransac_eps, 99, correctValues);
    createTrackbar("Success probability (%):",   "RANSAC Parameters", &ransac_prob, 99, correctValues);

    createTrackbar("H Min:", "HSV Thresholds", &h_min, 255, nullptr);
    createTrackbar("H Max:", "HSV Thresholds", &h_max, 255, nullptr);
    createTrackbar("V Min:", "HSV Thresholds", &v_min, 255, nullptr);
    createTrackbar("V Max:", "HSV Thresholds", &v_max, 255, nullptr);
    createTrackbar("S Min:", "HSV Thresholds", &s_min, 255, nullptr);
    createTrackbar("S Max:", "HSV Thresholds", &s_max, 255, nullptr);

    createTrackbar("B Min:", "BGR Thresholds", &b_min, 255, nullptr);
    createTrackbar("B Max:", "BGR Thresholds", &b_max, 255, nullptr);
    createTrackbar("G Min:", "BGR Thresholds", &g_min, 255, nullptr);
    createTrackbar("G Max:", "BGR Thresholds", &g_max, 255, nullptr);
    createTrackbar("R Min:", "BGR Thresholds", &r_min, 255, nullptr);
    createTrackbar("R Max:", "BGR Thresholds", &r_max, 255, nullptr);
}
// reading an writing from/to the calibration files ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void readCalib() {
    string name;
    int value;

    bool set_bw_tresh(false), set_voting_binsize(false), set_MOG_learning(false);

    bool set_accumulator_res(false), set_min_dist(false), set_canny_tresh(false), set_center_tresh(false), set_min_radius(false), set_max_radius(false);
    bool set_roi_upper(false), set_roi_lower(false), set_roi_left(false), set_roi_right(false);
    bool set_ransac_eps(false), set_ransac_prob(false), set_ransac_thresh(false), set_ransac_size(false);

    ifstream inputFile(inputName);

    if (!inputFile.is_open()) {
        cout << "Input calibration file not readable. ";
        setDefaults();
        return;
    }

    while (!inputFile.eof()) {
        inputFile >> name >> value;
        // cout << name << ": " << value << endl;

        if (name == string("bw_tresh") && !set_bw_tresh) {
            bw_tresh = value;
            set_bw_tresh = true;        
        }        
        if (name == string("canny_tresh") && !set_canny_tresh) {
            canny_tresh = value;
            set_canny_tresh = true;
        }
        if (name == string("center_tresh") && !set_center_tresh) {
            center_tresh = value;
            set_center_tresh = true;
        }
        if (name == string("min_dist") && !set_min_dist) {
            min_dist = value;
            set_min_dist = true;
        }
        if (name == string("min_radius") && !set_min_radius) {
            min_radius = value;
            set_min_radius = true;
        }
        if (name == string("max_radius") && !set_max_radius) {
            max_radius = value;
            set_max_radius = true;    
        }
        if (name == string("accumulator_res") && !set_accumulator_res) {
            accumulator_res = value;
            set_accumulator_res = true;    
        }
        if (name == string("roi_upper") && !set_roi_upper) {
            roi_upper = value;
            set_roi_upper = true;
        }
        if (name == string("roi_lower") && !set_roi_lower) {
            roi_lower = value;
            set_roi_lower = true;
        }
        if (name == string("roi_left") && !set_roi_left) {
            roi_left = value;
            set_roi_left = true;
        }
        if (name == string("roi_right") && !set_roi_right) {
            roi_right = value;
            set_roi_right = true;
        }
        if (name == string("voting_binsize") && !set_voting_binsize) {
            voting_binsize = value;
            set_voting_binsize = true;
        }
        if (name == string("ransac_size") && !set_ransac_size) {
            ransac_size = value;
            set_ransac_size = true;
        }
        if (name == string("ransac_eps") && !set_ransac_eps) {
            ransac_eps = value;
            set_ransac_eps = true;
        }
        if (name == string("ransac_prob") && !set_ransac_prob) {
            ransac_prob = value;
            set_ransac_prob = true;
        }
        if (name == string("ransac_thresh") && !set_ransac_thresh) {
            ransac_thresh = value;
            set_ransac_thresh = true;
        }
        if (name == string("MOG_learning") && !set_MOG_learning) {
            MOG_learningrate = value;
            set_MOG_learning = true;
        }
    }

    inputFile.close();

    bool hough_set(set_accumulator_res && set_min_dist && set_canny_tresh && set_center_tresh && set_min_radius && set_max_radius);
    bool roi_set(set_roi_upper && set_roi_lower && set_roi_left && set_roi_right);
    bool ransac_params_set(set_ransac_eps && set_ransac_prob && set_ransac_thresh && set_ransac_size);
    
    bool all_set(set_bw_tresh && set_voting_binsize && hough_set && roi_set && ransac_params_set && set_MOG_learning);


    if (all_set) {
        cout << "Succesfully read calibration parameters from file." << endl;
    } else {
        cout << "There was an error during the import of the calibration parameters. ";
        setDefaults();
    }
}
void setDefaults() {
    cout << "Setting default values for parameters." << endl;

    roi_upper = cam_height*1./7.2;
    roi_lower = cam_height*1./6;
    roi_left  = cam_width*1./2.1;
    roi_right = cam_width*1./2.5;
    bw_tresh = 140;
    canny_tresh = 100;
    center_tresh = 8;
    accumulator_res = 1;
    min_dist = 50;
    min_radius = 15;
    max_radius = 25;
    voting_binsize = 5;
    MOG_learningrate = 10;

    using namespace cv::videostab;
    RansacParams params = RansacParams::translationMotionStd();  // standard parameter for translational motion model
    ransac_size = params.size,
    ransac_thresh = params.thresh*100;
    ransac_eps = params.eps*100;
    ransac_prob = params.prob*100;
}
void writeCalib() {
    ofstream outputFile(outputName);

    if (outputFile.is_open()) {
        outputFile << "roi_upper\t"       << roi_upper    << endl;
        outputFile << "roi_lower\t"       << roi_lower    << endl;
        outputFile << "roi_left\t"        << roi_left     << endl;
        outputFile << "roi_right\t"       << roi_right    << endl;
        outputFile << "bw_tresh\t"        << bw_tresh     << endl;
        outputFile << "canny_tresh\t"     << canny_tresh  << endl;
        outputFile << "center_tresh\t"    << center_tresh << endl;
        outputFile << "min_dist\t"        << min_dist     << endl;
        outputFile << "min_radius\t"      << min_radius   << endl;
        outputFile << "max_radius\t"      << max_radius   << endl;
        outputFile << "ransac_size\t"     << ransac_size     << endl;
        outputFile << "ransac_thresh\t"   << ransac_thresh   << endl;
        outputFile << "ransac_eps\t"      << ransac_eps      << endl;
        outputFile << "ransac_prob\t"     << ransac_prob     << endl;
        outputFile << "accumulator_res\t" << accumulator_res << endl;
        outputFile << "voting_binsize\t"  << voting_binsize  << endl;
        outputFile << "MOG_learning\t"    << MOG_learningrate << endl;
        outputFile.close();

        cout << "Succesfully written calibration parameters to file." << endl;
    } else {
        cout << "Unable to open output file." << endl;
    }
}
