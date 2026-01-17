#include "feature_tracker.h"

int FeatureTracker::n_id = 0;
using namespace cv;
int maxLevel = 3;
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<descriptors> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);       // 特征点id, 一开始给这些新的特征点赋值-1， 会在updateID()函数里用个全局变量给他赋值
        track_cnt.push_back(1);  // 初始化特征点的观测次数：1次
    }
}

cv::Mat distance(cv::Mat image)
{
	cv::Mat edge_left = image;
	cv::Mat dst1;
    Mat dist = image.clone();
	cv::distanceTransform(cv::Scalar(255) - edge_left, dst1, CV_DIST_L2, 3);  //距离变换 L2好用
	cv::normalize(dst1, dst1, 0, 255, CV_MINMAX); //为了显示清晰，做了0~255归一化
	float *p;
	uchar *q;
	for (int i = 0; i < dst1.rows; i++)
	{
		p = dst1.ptr<float>(i);//获取每行首地址
		q = dist.ptr<uchar>(i);//获取每行首地
		for (int j = 0; j < dst1.cols; ++j)
		{
			int temp = round(p[j]);
			q[j] = temp;
		}
	}
    //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(16, 16));
    //clahe->apply(dist, dist);
	return dist;
}
void calcOpticalFlowPyrLK2(const Mat& prevImg, const Mat& nextImg, vector<Mat> cur_Pyr, vector<Mat>& forw_Pyr,
       const vector<Point2f>& prevPts,
       vector<Point2f>& nextPts,
       vector<uchar>& status,
       vector<float>& err,
       Size winSize,
       int maxLevel,
       TermCriteria criteria,
       double derivLambda,
       int flags)
{
       derivLambda = std::min(std::max(derivLambda, 0.), 1.);
       double lambda1 = 1. - derivLambda, lambda2 = derivLambda;
       const int derivKernelSize = 3;
       const float deriv1Scale = 0.5f / 4.f;
       const float deriv2Scale = 0.25f / 4.f;
       const int derivDepth = CV_32F;
       Point2f halfWin((winSize.width - 1)*0.5f, (winSize.height - 1)*0.5f);
       CV_Assert(maxLevel >= 0 && winSize.width > 2 && winSize.height > 2);
       CV_Assert(prevImg.size() == nextImg.size() &&
              prevImg.type() == nextImg.type());
       size_t npoints = prevPts.size();
       nextPts.resize(npoints);
       status.resize(npoints);
       for (size_t i = 0; i < npoints; i++)
              status[i] = true;
       err.resize(npoints);
       if (npoints == 0)
            return;
       vector<Mat> prevPyr, nextPyr;
       int cn = prevImg.channels();
       prevPyr = cur_Pyr;
       //buildPyramid(prevImg, prevPyr, maxLevel);
       buildPyramid(nextImg, nextPyr, maxLevel);
       for (int level = maxLevel; level >= 0; level--)
       {
           Mat tmp = distance(nextPyr[level]);
           nextPyr[level] = tmp;
           forw_Pyr[level] = tmp;
       }
       // I, dI/dx ~ Ix, dI/dy ~ Iy, d2I/dx2 ~ Ixx, d2I/dxdy ~ Ixy, d2I/dy2 ~ Iyy
       Mat derivIBuf((prevImg.rows + winSize.height * 2),
              (prevImg.cols + winSize.width * 2),
              CV_MAKETYPE(derivDepth, cn * 6));
       // J, dJ/dx ~ Jx, dJ/dy ~ Jy
       Mat derivJBuf((prevImg.rows + winSize.height * 2),
              (prevImg.cols + winSize.width * 2),
              CV_MAKETYPE(derivDepth, cn * 3));
       Mat tempDerivBuf(prevImg.size(), CV_MAKETYPE(derivIBuf.type(), cn));
       Mat derivIWinBuf(winSize, derivIBuf.type());
       if ((criteria.type & TermCriteria::COUNT) == 0)
              criteria.maxCount = 30;
       else
              criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
       if ((criteria.type & TermCriteria::EPS) == 0)
              criteria.epsilon = 0.01;
       else
              criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
       criteria.epsilon *= criteria.epsilon;
       for (int level = maxLevel; level >= 0; level--)
       {
              int k;
              Size imgSize = prevPyr[level].size();
              Mat tempDeriv(imgSize, tempDerivBuf.type(), tempDerivBuf.data);
              Mat _derivI(imgSize.height + winSize.height * 2,
                     imgSize.width + winSize.width * 2,
                     derivIBuf.type(), derivIBuf.data);
              Mat _derivJ(imgSize.height + winSize.height * 2,
                     imgSize.width + winSize.width * 2,
                     derivJBuf.type(), derivJBuf.data);
              Mat derivI(_derivI, Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
              Mat derivJ(_derivJ, Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
              CvMat cvderivI = _derivI;
              cvZero(&cvderivI);
              CvMat cvderivJ = _derivJ;
              cvZero(&cvderivJ);
              vector<int> fromTo(cn * 2);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2] = k;
              prevPyr[level].convertTo(tempDeriv, derivDepth);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              // compute spatial derivatives and merge them together
              Sobel(prevPyr[level], tempDeriv, derivDepth, 1, 0, derivKernelSize, deriv1Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6 + 1;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              Sobel(prevPyr[level], tempDeriv, derivDepth, 0, 1, derivKernelSize, deriv1Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6 + 2;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              Sobel(prevPyr[level], tempDeriv, derivDepth, 2, 0, derivKernelSize, deriv2Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6 + 3;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              Sobel(prevPyr[level], tempDeriv, derivDepth, 1, 1, derivKernelSize, deriv2Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6 + 4;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              Sobel(prevPyr[level], tempDeriv, derivDepth, 0, 2, derivKernelSize, deriv2Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6 + 5;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              nextPyr[level].convertTo(tempDeriv, derivDepth);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 3;
              mixChannels(&tempDeriv, 1, &derivJ, 1, &fromTo[0], cn);
              Sobel(nextPyr[level], tempDeriv, derivDepth, 1, 0, derivKernelSize, deriv1Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 3 + 1;
              mixChannels(&tempDeriv, 1, &derivJ, 1, &fromTo[0], cn);
              Sobel(nextPyr[level], tempDeriv, derivDepth, 0, 1, derivKernelSize, deriv1Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 3 + 2;
              mixChannels(&tempDeriv, 1, &derivJ, 1, &fromTo[0], cn);
              /*copyMakeBorder( derivI, _derivI, winSize.height, winSize.height,

                     winSize.width, winSize.width, BORDER_CONSTANT );

              copyMakeBorder( derivJ, _derivJ, winSize.height, winSize.height,

                     winSize.width, winSize.width, BORDER_CONSTANT );*/
              for (size_t ptidx = 0; ptidx < npoints; ptidx++)
              {
                     Point2f prevPt = prevPts[ptidx] * (float)(1. / (1 << level));
                     Point2f nextPt;
                     if (level == maxLevel)
                     {
                            if (flags & OPTFLOW_USE_INITIAL_FLOW)
                                   nextPt = nextPts[ptidx] * (float)(1. / (1 << level));
                            else
                                   nextPt = prevPt;
                     }
                     else
                            nextPt = nextPts[ptidx] * 2.f;
                     nextPts[ptidx] = nextPt;
                     Point2i iprevPt, inextPt;
                     prevPt -= halfWin;
                     iprevPt.x = cvFloor(prevPt.x);
                     iprevPt.y = cvFloor(prevPt.y);
                     if (iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
                            iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows)
                     {
                            if (level == 0)
                            {
                                   status[ptidx] = false;
                                   err[ptidx] = FLT_MAX;
                            }
                            continue;
                     }
                     float a = prevPt.x - iprevPt.x;
                     float b = prevPt.y - iprevPt.y;
                     float w00 = (1.f - a)*(1.f - b), w01 = a * (1.f - b);
                     float w10 = (1.f - a)*b, w11 = a * b;
                     size_t stepI = derivI.step / derivI.elemSize1();
                     size_t stepJ = derivJ.step / derivJ.elemSize1();
                     int cnI = cn * 6, cnJ = cn * 3;
                     double A11 = 0, A12 = 0, A22 = 0;
                     double iA11 = 0, iA12 = 0, iA22 = 0;
                     // extract the patch from the first image
                     int x, y;
                     for (y = 0; y < winSize.height; y++)
                     {
                            const float* src = (const float*)(derivI.data +
                                   (y + iprevPt.y)*derivI.step) + iprevPt.x*cnI;
                            float* dst = (float*)(derivIWinBuf.data + y * derivIWinBuf.step);
                            for (x = 0; x < winSize.width*cnI; x += cnI, src += cnI)
                            {
                                   float I = src[0] * w00 + src[cnI] * w01 + src[stepI] * w10 + src[stepI + cnI] * w11;
                                   dst[x] = I;
                                   float Ix = src[1] * w00 + src[cnI + 1] * w01 + src[stepI + 1] * w10 + src[stepI + cnI + 1] * w11;
                                   float Iy = src[2] * w00 + src[cnI + 2] * w01 + src[stepI + 2] * w10 + src[stepI + cnI + 2] * w11;
                                   dst[x + 1] = Ix; dst[x + 2] = Iy;
                                   float Ixx = src[3] * w00 + src[cnI + 3] * w01 + src[stepI + 3] * w10 + src[stepI + cnI + 3] * w11;
                                   float Ixy = src[4] * w00 + src[cnI + 4] * w01 + src[stepI + 4] * w10 + src[stepI + cnI + 4] * w11;
                                   float Iyy = src[5] * w00 + src[cnI + 5] * w01 + src[stepI + 5] * w10 + src[stepI + cnI + 5] * w11;
                                   dst[x + 3] = Ixx; dst[x + 4] = Ixy; dst[x + 5] = Iyy;
                                   iA11 += (double)Ix*Ix;
                                   iA12 += (double)Ix*Iy;
                                   iA22 += (double)Iy*Iy;
                                   A11 += (double)Ixx*Ixx + (double)Ixy*Ixy;
                                   A12 += Ixy * ((double)Ixx + Iyy);
                                   A22 += (double)Ixy*Ixy + (double)Iyy*Iyy;
                            }
                     }
                     A11 = lambda1 * iA11 + lambda2 * A11;
                     A12 = lambda1 * iA12 + lambda2 * A12;
                     A22 = lambda1 * iA22 + lambda2 * A22;
                     double D = A11 * A22 - A12 * A12;
                     double minEig = (A22 + A11 - std::sqrt((A11 - A22)*(A11 - A22) +
                            4.*A12*A12)) / (2 * winSize.width*winSize.height);
                     err[ptidx] = (float)minEig;
                     if (D < DBL_EPSILON)
                     {
                            if (level == 0)
                                   status[ptidx] = false;
                            continue;
                     }
                     D = 1. / D;
                     nextPt -= halfWin;
                     Point2f prevDelta;
                     for (int j = 0; j < criteria.maxCount; j++)
                     {
                            inextPt.x = cvFloor(nextPt.x);
                            inextPt.y = cvFloor(nextPt.y);
                            if (inextPt.x < -winSize.width || inextPt.x >= derivJ.cols ||
                                   inextPt.y < -winSize.height || inextPt.y >= derivJ.rows)
                            {
                                   if (level == 0)
                                          status[ptidx] = false;
                                   break;
                            }
                            a = nextPt.x - inextPt.x;
                            b = nextPt.y - inextPt.y;
                            w00 = (1.f - a)*(1.f - b); w01 = a * (1.f - b);
                            w10 = (1.f - a)*b; w11 = a * b;
                            double b1 = 0, b2 = 0, ib1 = 0, ib2 = 0;
                            for (y = 0; y < winSize.height; y++)
                            {
                                   const float* src = (const float*)(derivJ.data +
                                          (y + inextPt.y)*derivJ.step) + inextPt.x*cnJ;
                                   const float* Ibuf = (float*)(derivIWinBuf.data + y * derivIWinBuf.step);
                                   for (x = 0; x < winSize.width; x++, src += cnJ, Ibuf += cnI)
                                   {
                                          double It = src[0] * w00 + src[cnJ] * w01 + src[stepJ] * w10 +
                                                 src[stepJ + cnJ] * w11 - Ibuf[0];
                                          double Ixt = src[1] * w00 + src[cnJ + 1] * w01 + src[stepJ + 1] * w10 +
                                                 src[stepJ + cnJ + 1] * w11 - Ibuf[1];
                                          double Iyt = src[2] * w00 + src[cnJ + 2] * w01 + src[stepJ + 2] * w10 +
                                                 src[stepJ + cnJ + 2] * w11 - Ibuf[2];
                                          b1 += Ixt * Ibuf[3] + Iyt * Ibuf[4];
                                          b2 += Ixt * Ibuf[4] + Iyt * Ibuf[5];
                                          ib1 += It * Ibuf[1];
                                          ib2 += It * Ibuf[2];
                                   }
                            }
                            b1 = lambda1 * ib1 + lambda2 * b1;
                            b2 = lambda1 * ib2 + lambda2 * b2;
                            Point2f delta((float)((A12*b2 - A22 * b1) * D),
                                   (float)((A12*b1 - A11 * b2) * D));
                            //delta = -delta;
                            nextPt += delta;
                            nextPts[ptidx] = nextPt + halfWin;
                            if (delta.ddot(delta) <= criteria.epsilon)
                                   break;
                            if (j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
                                   std::abs(delta.y + prevDelta.y) < 0.01)
                            {
                                   nextPts[ptidx] -= delta * 0.5f;
                                   break;
                            }
                            prevDelta = delta;
                     }
              }
       }
}

void calcOpticalFlowPyrLK1(const Mat& prevImg, const Mat& nextImg, 
       const vector<Point2f>& prevPts, 
       vector<Point2f>& nextPts, 
       vector<uchar>& status, // 
       vector<float>& err, //
       Size winSize, //
       int maxLevel,
       TermCriteria criteria,
       double derivLambda,
       int flags)
{
       derivLambda = std::min(std::max(derivLambda, 0.), 1.); //cut lambda within 0~1
       double lambda1 = 1. - derivLambda, lambda2 = derivLambda;
       const int derivKernelSize = 3;
       const float deriv1Scale = 0.5f / 4.f;
       const float deriv2Scale = 0.25f / 4.f;
       const int derivDepth = CV_32F;
       Point2f halfWin((winSize.width - 1)*0.5f, (winSize.height - 1)*0.5f);
       CV_Assert(maxLevel >= 0 && winSize.width > 2 && winSize.height > 2);
       CV_Assert(prevImg.size() == nextImg.size() &&
              prevImg.type() == nextImg.type());
       size_t npoints = prevPts.size();
       nextPts.resize(npoints);
       status.resize(npoints);
       for (size_t i = 0; i < npoints; i++)
              status[i] = true;
       err.resize(npoints);
       if (npoints == 0)
            return;
       vector<Mat> prevPyr, nextPyr;
       int cn = prevImg.channels();
       buildPyramid(prevImg, prevPyr, maxLevel);
       buildPyramid(nextImg, nextPyr, maxLevel);
       for (int level = maxLevel; level >= 0; level--)
       {
           Mat tmp = prevPyr[level];
           prevPyr[level] = distance(tmp);
           tmp = nextPyr[level];
           nextPyr[level] = distance(tmp);
       }
       // I, dI/dx ~ Ix, dI/dy ~ Iy, d2I/dx2 ~ Ixx, d2I/dxdy ~ Ixy, d2I/dy2 ~ Iyy
       Mat derivIBuf((prevImg.rows + winSize.height * 2),
              (prevImg.cols + winSize.width * 2),
              CV_MAKETYPE(derivDepth, cn * 6));
       // J, dJ/dx ~ Jx, dJ/dy ~ Jy
       Mat derivJBuf((prevImg.rows + winSize.height * 2),
              (prevImg.cols + winSize.width * 2),
              CV_MAKETYPE(derivDepth, cn * 3));
       Mat tempDerivBuf(prevImg.size(), CV_MAKETYPE(derivIBuf.type(), cn));
       Mat derivIWinBuf(winSize, derivIBuf.type());
       if ((criteria.type & TermCriteria::COUNT) == 0)
              criteria.maxCount = 30;
       else
              criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
       if ((criteria.type & TermCriteria::EPS) == 0)
              criteria.epsilon = 0.01;
       else
              criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
       criteria.epsilon *= criteria.epsilon;
       for (int level = maxLevel; level >= 0; level--)
       {
              int k;
              Size imgSize = prevPyr[level].size();
              Mat tempDeriv(imgSize, tempDerivBuf.type(), tempDerivBuf.data);
              Mat _derivI(imgSize.height + winSize.height * 2,
                     imgSize.width + winSize.width * 2,
                     derivIBuf.type(), derivIBuf.data);
              Mat _derivJ(imgSize.height + winSize.height * 2,
                     imgSize.width + winSize.width * 2,
                     derivJBuf.type(), derivJBuf.data);
              Mat derivI(_derivI, Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
              Mat derivJ(_derivJ, Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
              _derivI = Mat::zeros(_derivI.size(), _derivI.type());
              _derivJ = Mat::zeros(_derivJ.size(), _derivJ.type());
              vector<int> fromTo(cn * 2);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2] = k;
              prevPyr[level].convertTo(tempDeriv, derivDepth);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              // compute spatial derivatives and merge them together
              Sobel(prevPyr[level], tempDeriv, derivDepth, 1, 0, derivKernelSize, deriv1Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6 + 1;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              Sobel(prevPyr[level], tempDeriv, derivDepth, 0, 1, derivKernelSize, deriv1Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6 + 2;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              Sobel(prevPyr[level], tempDeriv, derivDepth, 2, 0, derivKernelSize, deriv2Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6 + 3;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              Sobel(prevPyr[level], tempDeriv, derivDepth, 1, 1, derivKernelSize, deriv2Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6 + 4;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              Sobel(prevPyr[level], tempDeriv, derivDepth, 0, 2, derivKernelSize, deriv2Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 6 + 5;
              mixChannels(&tempDeriv, 1, &derivI, 1, &fromTo[0], cn);
              nextPyr[level].convertTo(tempDeriv, derivDepth);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 3;
              mixChannels(&tempDeriv, 1, &derivJ, 1, &fromTo[0], cn);
              Sobel(nextPyr[level], tempDeriv, derivDepth, 1, 0, derivKernelSize, deriv1Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 3 + 1;
              mixChannels(&tempDeriv, 1, &derivJ, 1, &fromTo[0], cn);
              Sobel(nextPyr[level], tempDeriv, derivDepth, 0, 1, derivKernelSize, deriv1Scale);
              for (k = 0; k < cn; k++)
                     fromTo[k * 2 + 1] = k * 3 + 2;
              mixChannels(&tempDeriv, 1, &derivJ, 1, &fromTo[0], cn);
              /*copyMakeBorder( derivI, _derivI, winSize.height, winSize.height,

                     winSize.width, winSize.width, BORDER_CONSTANT );

              copyMakeBorder( derivJ, _derivJ, winSize.height, winSize.height,

                     winSize.width, winSize.width, BORDER_CONSTANT );*/
              for (size_t ptidx = 0; ptidx < npoints; ptidx++)
              {
                     Point2f prevPt = prevPts[ptidx] * (float)(1. / (1 << level));
                     Point2f nextPt;
                     if (level == maxLevel)
                     {
                            if (flags & OPTFLOW_USE_INITIAL_FLOW)
                                   nextPt = nextPts[ptidx] * (float)(1. / (1 << level));
                            else
                                   nextPt = prevPt;
                     }
                     else
                            nextPt = nextPts[ptidx] * 2.f;
                     nextPts[ptidx] = nextPt;
                     Point2i iprevPt, inextPt;
                     prevPt -= halfWin;
                     iprevPt.x = cvFloor(prevPt.x);
                     iprevPt.y = cvFloor(prevPt.y);
                     if (iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
                            iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows)
                     {
                            if (level == 0)
                            {
                                   status[ptidx] = false;
                                   err[ptidx] = FLT_MAX;
                            }
                            continue;
                     }
                     float a = prevPt.x - iprevPt.x;
                     float b = prevPt.y - iprevPt.y;
                     float w00 = (1.f - a)*(1.f - b), w01 = a * (1.f - b);
                     float w10 = (1.f - a)*b, w11 = a * b;
                     size_t stepI = derivI.step / derivI.elemSize1();
                     size_t stepJ = derivJ.step / derivJ.elemSize1();
                     int cnI = cn * 6, cnJ = cn * 3;
                     double A11 = 0, A12 = 0, A22 = 0;
                     double iA11 = 0, iA12 = 0, iA22 = 0;
                     // extract the patch from the first image
                     int x, y;
                     for (y = 0; y < winSize.height; y++)
                     {
                            const float* src = (const float*)(derivI.data +
                                   (y + iprevPt.y)*derivI.step) + iprevPt.x*cnI;
                            float* dst = (float*)(derivIWinBuf.data + y * derivIWinBuf.step);
                            for (x = 0; x < winSize.width*cnI; x += cnI, src += cnI)
                            {
                                   float I = src[0] * w00 + src[cnI] * w01 + src[stepI] * w10 + src[stepI + cnI] * w11;
                                   dst[x] = I;
                                   float Ix = src[1] * w00 + src[cnI + 1] * w01 + src[stepI + 1] * w10 + src[stepI + cnI + 1] * w11;
                                   float Iy = src[2] * w00 + src[cnI + 2] * w01 + src[stepI + 2] * w10 + src[stepI + cnI + 2] * w11;
                                   dst[x + 1] = Ix; dst[x + 2] = Iy;
                                   float Ixx = src[3] * w00 + src[cnI + 3] * w01 + src[stepI + 3] * w10 + src[stepI + cnI + 3] * w11;
                                   float Ixy = src[4] * w00 + src[cnI + 4] * w01 + src[stepI + 4] * w10 + src[stepI + cnI + 4] * w11;
                                   float Iyy = src[5] * w00 + src[cnI + 5] * w01 + src[stepI + 5] * w10 + src[stepI + cnI + 5] * w11;
                                   dst[x + 3] = Ixx; dst[x + 4] = Ixy; dst[x + 5] = Iyy;
                                   iA11 += (double)Ix*Ix;
                                   iA12 += (double)Ix*Iy;
                                   iA22 += (double)Iy*Iy;
                                   A11 += (double)Ixx*Ixx + (double)Ixy*Ixy;
                                   A12 += Ixy * ((double)Ixx + Iyy);
                                   A22 += (double)Ixy*Ixy + (double)Iyy*Iyy;
                            }
                     }
                     A11 = lambda1 * iA11 + lambda2 * A11;
                     A12 = lambda1 * iA12 + lambda2 * A12;
                     A22 = lambda1 * iA22 + lambda2 * A22;
                     double D = A11 * A22 - A12 * A12;
                     double minEig = (A22 + A11 - std::sqrt((A11 - A22)*(A11 - A22) +
                            4.*A12*A12)) / (2 * winSize.width*winSize.height);
                     err[ptidx] = (float)minEig;
                     if (D < DBL_EPSILON)
                     {
                            if (level == 0)
                                   status[ptidx] = false;
                            continue;
                     }
                     D = 1. / D;
                     nextPt -= halfWin;
                     Point2f prevDelta;
                     for (int j = 0; j < criteria.maxCount; j++)
                     {
                            inextPt.x = cvFloor(nextPt.x);
                            inextPt.y = cvFloor(nextPt.y);
                            if (inextPt.x < -winSize.width || inextPt.x >= derivJ.cols ||
                                   inextPt.y < -winSize.height || inextPt.y >= derivJ.rows)
                            {
                                   if (level == 0)
                                          status[ptidx] = false;
                                   break;
                            }
                            a = nextPt.x - inextPt.x;
                            b = nextPt.y - inextPt.y;
                            w00 = (1.f - a)*(1.f - b); w01 = a * (1.f - b);
                            w10 = (1.f - a)*b; w11 = a * b;
                            double b1 = 0, b2 = 0, ib1 = 0, ib2 = 0;
                            for (y = 0; y < winSize.height; y++)
                            {
                                   const float* src = (const float*)(derivJ.data +
                                          (y + inextPt.y)*derivJ.step) + inextPt.x*cnJ;
                                   const float* Ibuf = (float*)(derivIWinBuf.data + y * derivIWinBuf.step);
                                   for (x = 0; x < winSize.width; x++, src += cnJ, Ibuf += cnI)
                                   {
                                          double It = src[0] * w00 + src[cnJ] * w01 + src[stepJ] * w10 +
                                                 src[stepJ + cnJ] * w11 - Ibuf[0];
                                          double Ixt = src[1] * w00 + src[cnJ + 1] * w01 + src[stepJ + 1] * w10 +
                                                 src[stepJ + cnJ + 1] * w11 - Ibuf[1];
                                          double Iyt = src[2] * w00 + src[cnJ + 2] * w01 + src[stepJ + 2] * w10 +
                                                 src[stepJ + cnJ + 2] * w11 - Ibuf[2];
                                          b1 += Ixt * Ibuf[3] + Iyt * Ibuf[4];
                                          b2 += Ixt * Ibuf[4] + Iyt * Ibuf[5];
                                          ib1 += It * Ibuf[1];
                                          ib2 += It * Ibuf[2];
                                   }
                            }
                            b1 = lambda1 * ib1 + lambda2 * b1;
                            b2 = lambda1 * ib2 + lambda2 * b2;
                            Point2f delta((float)((A12*b2 - A22 * b1) * D),
                                   (float)((A12*b1 - A11 * b2) * D));
                            //delta = -delta;
                            nextPt += delta;
                            nextPts[ptidx] = nextPt + halfWin;
                            if (delta.ddot(delta) <= criteria.epsilon)
                                   break;
                            if (j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
                                   std::abs(delta.y + prevDelta.y) < 0.01)
                            {
                                   nextPts[ptidx] -= delta * 0.5f;
                                   break;
                            }
                            prevDelta = delta;
                     }
              }
       }
}

cv::Mat AutoCanny(cv::Mat mImEH)
{
    Mat SmoothedImage;
    GaussianBlur( mImEH, SmoothedImage, Size( 3, 3 ), 1.8, 1.8 );

    /// Detect edges using canny (auto th)
    Mat dx, dy, dxM, dyM, histImg, hist;
    float th1;

    Sobel(SmoothedImage, dx, CV_16SC1, 1, 0, 3, 1, 0, BORDER_REPLICATE);
    Sobel(SmoothedImage, dy, CV_16SC1, 0, 1, 3, 1, 0, BORDER_REPLICATE);
    double maxMagGf = 0;
    cv::resize(dx, dxM,Size(0,0),0.2,0.2, INTER_NEAREST);
    cv::resize(dy, dyM,Size(0,0),0.2,0.2, INTER_NEAREST);
    add(abs(dxM),abs(dyM),SmoothedImage);
    SmoothedImage.convertTo(histImg, CV_32FC1);
    minMaxIdx(histImg, NULL, &maxMagGf);

    float bin = 256;
    calcHist(vector<Mat>{histImg},vector<int>{0},noArray()
             ,hist,vector<int>{bin},vector<float>{0,maxMagGf});
    int sum = 0, total = SmoothedImage.cols*SmoothedImage.rows*0.9;
    float ii = 0;
    for(; ii<bin; ii++)
    {
        sum+=hist.at<float>(ii);
        if(sum>total)
            break;
    }
    if(ii == bin)
        ii--;
    th1 = (ii+1)/bin*maxMagGf;
    cv::Mat mEdgeImage;
    Canny( dx, dy, mEdgeImage, th1, 0.9 * th1 ); //OpenCV3

    return mEdgeImage;
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time, Matrix3d relative_R, const sensor_msgs::PointCloudConstPtr& edgedes_msg,int init_flag) 
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
    if (EQUALIZE)   // 直方图均衡化
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;
    //img = AutoCanny(img);
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
        //cur_dst = forw_dst = distance(img);  
        cur_num =forw_num = edgedes_msg->points.size();
        // buildPyramid(forw_img, forw_Pyr, 3);
        // for (int level = 3; level >= 0; level--)
        // {
        //     Mat tmp = forw_Pyr[level];
        //     forw_Pyr[level] = distance(tmp);
        // }

       // vector<Mat> nextPyr;
       // buildPyramid(forw_img, nextPyr, 3);
       // forw_Pyr = cur_Pyr = nextPyr;
       // for (int level = maxLevel; level >= 0; level--)
       // {
       //     Mat tmp = nextPyr[level];
       //     forw_Pyr[level] = distance(tmp);
       //     cur_Pyr[level] = forw_Pyr[level].clone();
       // }
    }
    else
    {
        forw_img = img;
        //forw_dst = distance(img);
        forw_num = edgedes_msg->points.size();
    }
    // cout<<forw_num<<endl;
    forw_pts.clear();
    for_des.clear();
    if (cur_pts.size() > 0)       // i时刻的 特征点
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        TermCriteria  criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
        int flags = 1;
        double minEigThreshold = 1e-4;
        // 光流跟踪的结果放在 forw_pts
        // predictPtsInNextFrame(relative_R);
        // //std::cout<<relative_R++-<<std::endl;
        // forw_pts = predict_pts;
        // cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3,
        //                          cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        //                          cv::OPTFLOW_USE_INITIAL_FLOW);
        //cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
        //calcOpticalFlowPyrLK1(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3,criteria,minEigThreshold,flags);
        //if(abs(cur_num-forw_num)/forw_num<0.2)
        Eigen::AngleAxisd rotation_vector(relative_R);
        double temp_angle = rotation_vector.angle() * (180 / 3.1415) ;
        // cout<<temp_angle<<endl;
       //  if(init_flag == 0)
       //      cout<<00000000000000000000000<<endl;
       //  for (int level = maxLevel; level >= 0; level--)
       //  {
       //        cur_Pyr[level] = forw_Pyr[level].clone();
       //  }
        if(((temp_angle<0.2)&&abs(cur_num-forw_num)/cur_num<0.2)||init_flag==0)
       //if(((temp_angle<2)&&abs(cur_num-forw_num)/cur_num<0.2))
       //  if(abs(cur_num-forw_num)/forw_num<0.2)        
        {
            calcOpticalFlowPyrLK1(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3,criteria,minEigThreshold,flags);
            //calcOpticalFlowPyrLK2(cur_img, forw_img, cur_Pyr, forw_Pyr, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3,criteria,minEigThreshold,flags);
            cout<<1<<endl;
        }
        else
        {
            cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
            cout<<0<<endl;
       //      vector<Mat> nextPyr;
       //      buildPyramid(forw_img, nextPyr, 3);
       //      for (int level = maxLevel; level >= 0; level--)
       //      {
       //        Mat tmp = nextPyr[level];
       //        forw_Pyr[level] = distance(tmp);
       //      }
        }
        for (int i = 0; i < int(forw_pts.size()); i++)
        {
            if (status[i] && !inBorder(forw_pts[i]))    // 跟踪成功，但是在图像外的点，设置成跟踪失败
                status[i] = 0;
        }    
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
       //  reduceVector(for_des, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    if (PUB_THIS_FRAME)
    {
        rejectWithF();              // 通过计算F矩阵排除outlier

        for (auto &n : track_cnt)   // 对tracking上的特征的跟踪帧数进行更新，从第i帧成功跟踪到了i+1帧，跟踪帧数+1
            n++;

        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();                 // 设置模板，把那些已经检测出特征点的区域给掩盖掉, 其他区域用于检测新的特征点
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());  // 如果 当前特征点数目< MAX_CNT,  那就检测一些新的特征点
        if (n_max_cnt > 0)    // 少于最大特征点数目，那就补充新特征的
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask,3);  // 补充一些新的特征点
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();         // 将这个新的特征点加入 到 forw_pts
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
        prev_img = forw_img;
        prev_pts = forw_pts;
    }
    cur_dst = forw_dst;
    cur_img = forw_img;
    cur_pts = forw_pts;
//     cur_des = for_des;
    cur_num = forw_num;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_prev_pts(prev_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < prev_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_prev_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = prev_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;   // n_id 是个全局变量，给每个特征点一个独特的id
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(1);
}

void FeatureTracker::showUndistortion()
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    cv::Mat undist_map1_, undist_map2_;

    m_camera->initUndistortRectifyMap(undist_map1_,undist_map2_);
    cv::remap(cur_img, undistortedImg, undist_map1_, undist_map2_, CV_INTER_LINEAR);

    cv::imshow("undist", undistortedImg);
    cv::waitKey(1);
}

vector<cv::Point2f> FeatureTracker::undistortedPoints()
{
    vector<cv::Point2f> un_pts;
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }

    return un_pts;
}

void FeatureTracker::predictPtsInNextFrame(Matrix3d relative_R) {
    predict_pts.resize(cur_pts.size());
    for (int i = 0; i < cur_pts.size(); ++i) {
        Eigen::Vector3d tmp_P;
        m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_P);
        Eigen::Vector3d predict_P = relative_R * tmp_P;
        Eigen::Vector2d tmp_p;
        m_camera->spaceToPlane(predict_P, tmp_p);
        predict_pts[i].x = tmp_p.x();
        predict_pts[i].y = tmp_p.y();
    }
}