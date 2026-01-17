#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <sensor_msgs/PointCloud.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"
#include  <utility>
using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);
struct descriptors
{
  vector<int>des_valid;
  vector<double>dist;
  vector<vector<double>>des;
};
void reduceVector(vector<descriptors> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img);
    void readImage(const cv::Mat &_img, double _cur_time, Matrix3d relative_R);
    void readImage(const cv::Mat &_img, double _cur_time, Matrix3d relative_R,const sensor_msgs::PointCloudConstPtr &edgedes_msg);
    void readImage(const cv::Mat &_img, double _cur_time, Matrix3d relative_R,const sensor_msgs::PointCloudConstPtr &edgedes_msg,int init_flag);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);
    void showUndistortion();
    void predictPtsInNextFrame(Matrix3d relative_R);

    void rejectWithF();

    vector<cv::Point2f> undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img; //  prev : i-1 时刻，  cur: i 时刻， forw： i+1时刻
    cv::Mat cur_dst, forw_dst; 
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, predict_pts,forw_pts;
    vector<descriptors> cur_des,for_des;
    vector<int> ids;                     //  每个特征点的id
    vector<int> track_cnt;               //  记录某个特征已经跟踪多少帧了，即被多少帧看到了
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;
    static int n_id;
    int cur_num,forw_num;
    vector<cv::Mat>cur_Pyr,forw_Pyr;
};
