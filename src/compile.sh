
 #Compile under the src path
g++ -g -o yolov8-p2_thread main_test_threadpool.cc rkYolov8n.cc postprocess.cc resize_function.cc rknn_utils.cc preprocess.cc coreNum.cc -L../libs/librknn_api/lib -lrknnrt -I../libs/librknn_api/include -L../3rdparty/GCC_7_5/opencv/opencv-linux-aarch64/lib -lopencv_world -I../3rdparty/GCC_7_5/opencv/opencv-linux-aarch64/include -I../include -I/home/firefly/ljh/ItcMultiDetect_yolov8/src -L../3rdparty/GCC_7_5/rga/RK3588/lib/Linux/aarch64 -lrga -I../3rdparty/GCC_7_5/rga/RK3588/include -pthread

#添加rknn opencv路径到环境变量
#export LD_LIBRARY_PATH=/home/firefly/ljh/ItcMultiDetect/libs/librknn_api/lib:/home/firefly/ljh/ItcMultiDetect/3rdparty/GCC_7_5/opencv/opencv-linux-aarch64/lib:$LD_LIBRARY_PATH
