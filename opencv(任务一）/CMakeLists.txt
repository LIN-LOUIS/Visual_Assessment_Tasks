cmake_minimum_required(VERSION 2.8) # 选择最匹配的最低的Cmake版本
project( opencv1 ) # 创建的程序的名称（项目的名称）
find_package( OpenCV REQUIRED ) # 找到需求的库或者组件
include_directories( ${OpenCV_INCLUDE_DIRS} ) # 添加一个用于包含的文件去构建的文件夹
add_executable( opencv1 main.cpp ) # 添加参与编译的项目完整包
target_link_libraries( opencv1 ${OpenCV_LIBS} ) # 链接的库
