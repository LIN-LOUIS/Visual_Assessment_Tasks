/*
不使用 cv 提供的库函数，用三种方式实现图像的二值化（提示：遍历 cv::Mat 有几种方式？

*/



#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
//1.使用指针遍历 cv::Mat
void bin1(Mat& img, int threshold) {
    if (img.empty() || img.type() != CV_8UC1) {
        return; // 确保图像是有效的灰度图像
    }
    for (int i = 0; i < img.rows; ++i) {
        uchar* row_ptr = img.ptr<uchar>(i);  // 获取每一行的指针
        for (int j = 0; j < img.cols; ++j) {
            if (row_ptr[j] > threshold) {
                row_ptr[j] = 255;  // 大于阈值设为255
            } else {
                row_ptr[j] = 0;    // 否则设为0
            }
        }
    }
}
//2.使用 at() 方法遍历
void bin2(Mat& img, int threshold) {
    if (img.empty() || img.type() != CV_8UC1) {
        return; // 确保图像是有效的灰度图像
    }
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            uchar pixel_value = img.at<uchar>(i, j);
            img.at<uchar>(i, j) = (pixel_value > threshold) ? 255 : 0;
        }
    }
}
void bin3(cv::Mat& img, int threshold) {
    if (img.empty() || img.type() != CV_8UC1) {
        return; // 确保图像是有效的灰度图像
    }
    MatIterator_<uchar> it, end;
    for (it = img.begin<uchar>(), end = img.end<uchar>(); it != end; ++it) {
        *it = (*it > threshold) ? 255 : 0;
    }