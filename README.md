# yolov8s-rknn-multi-thread-inference
yolov8s在rk3588的推理部署demo，使用多线程池并行npu推理加速。只支持rk3588。

# Export onnx model
参照rk https://github.com/airockchip/ultralytics_yolov8 导出onnx模型,再转换成rk模型

# Inference
参照 https://github.com/leafqycc/rknn-cpp-Multithreading 代码进行修改，支持多线程池推理

# Note
1.main_test_threadpool.cc文件中可通过修改threadNum变量值来设定线程数量

2.coreNum.cc中可设定使用的npu核数

3.demo部署的是定制训练的yolov8s-p2检测模型，对应着四个输出尺度。如需部署原始的yolov8s,后处理代码参照https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8/cpp


# Reference
https://github.com/leafqycc/rknn-cpp-Multithreading

https://github.com/airockchip/ultralytics_yolov8

https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8/cpp
