#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#include <opencv2/opencv.hpp>

int main() {
    // 创建 VideoCapture 对象，用于打开摄像头
    VideoCapture cap(0);
    
    // 检查摄像头是否成功打开
    if (!cap.isOpened()) {
        std::cout << "无法打开摄像头！" << std::endl;
        return -1;
    }

    // 创建一个窗口，用于显示摄像头画面
    namedWindow("Camera", WINDOW_AUTOSIZE);

    // 用于存储摄像头捕获的帧
    Mat frame, grayFrame;
    
    // 计数器，控制图像转换为灰度的频率
    int frameCount = 0;
    const int grayFrequency = 10; // 每10帧转换一次为灰度图

    // 不断从摄像头读取画面并显示，直到用户按下 ESC 键退出
    while (true) {
        // 从摄像头捕获一帧
        cap >> frame;

        // 检查是否成功捕获到帧
        if (frame.empty()) {
            std::cout << "无法捕获帧！" << std::endl;
            break;
        }

        // 每隔一定帧数将图像转换为灰度图
        if (frameCount % grayFrequency == 0) {
            cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
            imshow("Camera", grayFrame); // 显示灰度图
        } else {
            imshow("Camera", frame); // 显示彩色图
        }

        // 计数器递增
        frameCount++;

        // 等待 30 毫秒，看用户是否按下 ESC 键 (ASCII码 27)
        if (waitKey(30) == 27) {
            std::cout << "ESC 键按下，退出！" << std::endl;
            break;
        }
    }

    // 释放摄像头
    cap.release();

    // 销毁窗口
    destroyAllWindows();

    return 0;
}
