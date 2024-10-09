#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov8n.h"
#include "rknnPool.hpp"



int main(int argc, char **argv)
{
    // char *model_name = NULL;
    // if (argc != 3)
    // {
    //     printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
    //     return -1;
    // }
    // // 参数二，模型所在路径/The path where the model is located
    // model_name = (char *)argv[1];
    // // 参数三, 视频/摄像头
    // char *vedio_name = argv[2];

    // 初始化rknn线程池/Initialize the rknn thread pool
    int threadNum = 1;
    std::string model_name = "/home/firefly/ljh/ItcMultiDetect_yolov8/models/yolov8s-p2-19classes-250epoch.rknn";
    rknnPool<rkYolov8n, cv::Mat, cv::Mat> testPool(model_name, threadNum);
    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    cv::namedWindow("Camera FPS");
    // cv::VideoCapture capture("/home/firefly/ljh/ItcMultiDetect_yolov8/output_video.mp4");
    cv::VideoCapture capture(11,cv::CAP_V4L2);
    // cv::VideoCapture capture("rtsp://172.16.40.84:553/live", cv::CAP_FFMPEG);

    int video_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int video_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int video_fps = capture.get(cv::CAP_PROP_FPS);
    int video_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
    printf("video info: width=%d, height=%d, fps=%d, frames=%d\n", video_width, video_height, video_fps, video_frames);
    if(!capture.isOpened()){
        std::cout << "video capture fail" << std::endl;
    }

    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    int frames = 0;
    auto beforeTime = startTime;
    while (capture.isOpened())
    {
        cv::Mat img;
        if (capture.read(img) == false)
            break;
        if (testPool.put(img) != 0)
            break;

        if (frames >= threadNum && testPool.get(img) != 0)
            break;
        
        cv::imshow("Camera FPS", img);
        if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
            break;
        frames++;

        if (frames % 60 == 0)
        {
        gettimeofday(&time, nullptr);
        auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
        printf("60帧内平均帧率:\t %f fps/s\n", 60.0 / float(currentTime - beforeTime) * 1000.0);
        // std::cout << "infer time: " << float(currentTime - beforeTime) << " ms" << std::endl;
        beforeTime = currentTime;
        }
    }

    // 清空rknn线程池/Clear the thread pool
    while (true)
    {
        cv::Mat img;
        if (testPool.get(img) != 0)
            break;
        cv::imshow("Camera FPS", img);
        if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
            break;
        frames++;
    }

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    return 0;
}