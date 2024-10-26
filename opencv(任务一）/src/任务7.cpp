#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat gammaCorrection(const Mat& src, float gamma) {
    // 创建查找表，查找表的大小是256（每个灰度级的像素值对应一个映射值）
    Mat lut(1, 256, CV_8UC1);
    uchar* p = lut.ptr();

    for (int i = 0; i < 256; i++) {
        // 应用伽马公式，将每个像素值映射到[0,255]范围内
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    // 应用查找表实现伽马矫正
    Mat dst;
    LUT(src, lut, dst);

    return dst;
}
int main() {
    //获取图像
    Mat image = imread("../image/伽马矫正.png");
    if (image.empty()) {
        cout << "无法读取图像" << endl;
        return -1;
    }
    // 设定伽马值（可以调整伽马值观察效果）
    float gamma = 0.5;

    // 执行伽马矫正
    Mat dst = gammaCorrection(image, gamma);

    // 显示结果
    imshow("Original Image", image);
    imshow("Gamma Corrected Image", dst);

    waitKey(12000);
    return 0;
}
