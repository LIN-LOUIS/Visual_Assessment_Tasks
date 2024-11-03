#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
    //获取图像
    Mat image = imread("../image/口罩.jpg");
    if (image.empty()) {
        cout << "无法读取图像" << endl;
        return -1;
    }
    // 缩小为原图的50%
    double scale = 0.5;  
    cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_LINEAR);

    // 转换为灰度图像
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    //二值化
    Mat edges;
    threshold(gray, edges, 0, 255,  THRESH_OTSU);
    //定义一个 10x10 的椭圆结构元素
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
    // 开运算操作
    morphologyEx(edges, edges, MORPH_OPEN, element);
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // 在原图上绘制轮廓的边界框
    int i=0;
    cv::Mat result = image.clone(); // 创建一个克隆图像以便绘制

    // 创建一个容器来存储凸包的点
    std::vector<cv::Point> hull;
    // 过滤后的凸包集合
    std::vector<std::vector<cv::Point>> filteredHulls;
    // 对每个轮廓计算凸包
    for (size_t i = 0; i < contours.size(); i++) {
        
        // 计算凸包
        cv::convexHull(contours[i], hull);
        // 计算凸包面积
        double area = cv::contourArea(hull);

        // 如果面积大于阈值，添加到过滤后的集合中
        if (area>100000) {
            filteredHulls.push_back(hull);
        }
    }



        cv::Scalar color = cv::Scalar(0, 255, 255); // 设置边框颜色为黄色
        
        cv::polylines(result, filteredHulls, true, color, 2); // 绘制凸包的边界线
        

    
    ////这4个点应该是图像中待变换区域的角点
    std::vector<cv::Point2f> srcPoints(4);
    //// 初始化最左、最右、最上和最下的点索引
    //int minLeft = 0, maxRight = 0, minTop = 0, maxBottom = 0;
    //for(const auto& hull1 : filteredHulls){
    //    for (size_t i = 1; i < hull1.size(); ++i) {
    //        if(hull1[i].x==0 || hull1[i].y==0) continue;       // 找到最左边的点
    //        if (hull1[i].x < hull1[minLeft].x) minLeft = i;       // 找到最左边的点
    //        if (hull1[i].x > hull1[maxRight].x) maxRight = i;     // 找到最右边的点
    //        if (hull1[i].y < hull1[minTop].y) minTop = i;         // 找到最上边的点
    //        if (hull1[i].y > hull1[maxBottom].y) maxBottom = i;   // 找到最下边的点
    //    }
    //  
    //// 将找到的 4 个极值点添加到 srcPoints 中，转为 Point2f 类型
    //srcPoints[0] = cv::Point2f(hull1[minLeft].x, hull1[minLeft].y);       // 左边
    //srcPoints[1] = cv::Point2f(hull1[maxRight].x, hull1[maxRight].y);     // 右边
    //srcPoints[2] = cv::Point2f(hull1[minTop].x, hull1[minTop].y);         // 上边
    //srcPoints[3] = cv::Point2f(hull1[maxBottom].x, hull1[maxBottom].y);   // 下边
    //}
    //std::cout<< std::endl;
    //  cv::Scalar color2 = cv::Scalar(0,0,0); // 设置边框颜色为黑色
    //    for (const auto& point : srcPoints) {
    //        cv::circle(result, point,true, color2, 2);
    //        std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
    //    }
    // 目标点可以是一个新的平面区域
    for(const auto& points : filteredHulls){ 
        double maxArea = 0.0;
        // 四重循环，遍历所有四点组合
        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = i + 1; j < points.size(); ++j) {
                for (size_t k = j + 1; k < points.size(); ++k) {
                    for (size_t l = k + 1; l < points.size(); ++l) {
                        // 当前四个点
                        std::vector<cv::Point> quad = { points[i], points[j], points[k], points[l] };

                        // 计算四边形的面积
                        double area = cv::contourArea(quad);

                        // 更新最大面积和顶点
                        if (area > maxArea) {
                            maxArea = area;
                            srcPoints[0] = cv::Point2f(points[i].x, points[i].y);     // 左边
                            srcPoints[1] = cv::Point2f(points[j].x, points[j].y);     // 右边
                            srcPoints[2] = cv::Point2f(points[k].x, points[k].y);     // 上边
                            srcPoints[3] = cv::Point2f(points[l].x, points[l].y);     // 下边
                        }
                    }
                }
            }
        }
    }
    std::vector<cv::Point2f> dstPoints = {
        {0, 0},            // 左上角
        {1000, 0},          // 右上角
        {1000, 600},        // 右下角
        {0, 600}           // 左下角
    };

    // 计算透视变换矩阵
    cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);

    // 设定目标图像的大小，这里宽为300，高为400（与目标点设置匹配）
    cv::Mat result1;
    cv::warpPerspective(image, result1, M, cv::Size(1000, 600));

    // 显示结果
    cv::imshow("Original", image);
    cv::imshow("Gray",edges);
    cv::imshow("Canny", result);
    cv::imshow("Bounding Boxes", result1);
    cv::waitKey(0);

    return 0;
}
