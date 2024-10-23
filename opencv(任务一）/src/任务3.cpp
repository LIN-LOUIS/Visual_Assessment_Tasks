#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#include <opencv2/opencv.hpp>

int main() {
    // 读取图像
    Mat image = imread("../image/回形针.png");
    if (image.empty()) {
        cout << "无法读取图像" << endl;
        return -1;
    }
    // 转换为灰度图像
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    //二值化
    Mat edges;
    threshold(gray, edges, 0, 255,  THRESH_OTSU);
    // 执行 Canny 边缘检测
    cv::Canny(edges, edges, 100, 200);
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 在原图上绘制轮廓的边界框
    int i=0;
    cv::Mat result = image.clone(); // 创建一个克隆图像以便绘制
    for (const auto& contour : contours) {
        // 计算轮廓的最小外接矩形
        cv::RotatedRect minAreaRect = cv::minAreaRect(contour);
        // 获取矩形的四个顶点
        cv::Point2f rectPoints[4];
        minAreaRect.points(rectPoints);
        // 绘制最小外接矩形
        for (int j = 0; j < 4; j++) {
            cv::line(result, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        // 在矩形的左上角添加文本标记
        i++;
        string text=to_string(i);
        cv::putText(result, text, rectPoints[0], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0,255), 3);
    }
    //保存结果
    cv::imwrite("../image/回形针结果.png", result);
    // 显示结果
    cv::imshow("Original", image);
    cv::imshow("Edges", edges);
    cv::imshow("Bounding Boxes", result);
    cv::waitKey(0);

    return 0;
}