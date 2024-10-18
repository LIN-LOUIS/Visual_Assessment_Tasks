#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#include <opencv2/opencv.hpp>

int main() {
    // 读取图像
    Mat image = imread("../image/image1.jpg",IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // 应用高斯模糊以减少噪声
    Mat blurred_image;
    GaussianBlur(image, blurred_image, Size(5, 5), 1.5);

    // 应用 Canny 边缘检测
    Mat edges;
    Canny(blurred_image, edges, 50, 150);

    //改变图像大小
    resize(edges, edges, Size(500, 500));
    resize(image, image, Size(500, 500));

    // 显示原图和边缘检测后的图像
      ::imshow("Original Image", image);
      ::imshow("Canny Edges", edges);

    // 等待按键按下，关闭窗口
      ::waitKey(0);
      ::destroyAllWindows();

    return 0;
}