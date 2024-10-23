#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#include <opencv2/opencv.hpp>

int main() {
    // 读取图像
    Mat image = imread("../image/腐蚀、膨胀、开运算、闭运算.png",IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "无法读取图像" << endl;
        return -1;
    }
    Mat ero;
    Mat dil;
    Mat open;
    Mat close;
    
    // 定义一个 10x10 的矩形结构元素
    Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
    
    //定义一个 10x10 的椭圆结构元素
    Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));

    //定义一个 10x10 的十字形结构元素
    Mat element3 = getStructuringElement(MORPH_CROSS, Size(10, 10));
    // 腐蚀操作
    erode(image, ero, element);
    erode(image, ero, element2);
    erode(image, ero, element3);
    
    //膨胀操作
    dilate(image, dil, element);
    dilate(image, dil, element2);
    dilate(image, dil, element3);

    // 开运算操作
    morphologyEx(image, open, MORPH_OPEN, element);
    morphologyEx(image, open, MORPH_OPEN, element2);
    morphologyEx(image, open, MORPH_OPEN, element3);

    // 闭运算操作
    morphologyEx(image, close, MORPH_CLOSE, element);
    morphologyEx(image, close, MORPH_CLOSE, element2);
    morphologyEx(image, close, MORPH_CLOSE, element3);

    // 显示结果
    imshow("原本的图像", image);
    imshow("腐蚀后的图像(矩形)", ero);
    imshow("膨胀后的图像(矩形)", dil);
    imshow("开运算后的图像(矩形)", open);
    imshow("闭运算后的图像(矩形)", close);

    imshow("腐蚀后的图像(椭圆)", ero);
    imshow("膨胀后的图像(椭圆)", dil);
    imshow("开运算后的图像(椭圆)", open);
    imshow("闭运算后的图像(椭圆)", close);

    imshow("腐蚀后的图像(十字形)", ero);
    imshow("膨胀后的图像(十字形)", dil);
    imshow("开运算后的图像(十字形)", open);
    imshow("闭运算后的图像(十字形)", close);
    waitKey(0);
    return 0;
}