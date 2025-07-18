/*
 * Copyright (c) 2025 Zhao Jiabing
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * @file yolov8_detector.h
 * @brief YOLOv8圆形检测器头文件
 * @author Generated by Copilot
 * @date 2025-07-13
 */

#ifndef YOLOV8_DETECTOR_H
#define YOLOV8_DETECTOR_H

#include <opencv2/opencv.hpp>
#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif
#include <vector>
#include <string>
#include <memory>

namespace yolov8 {

/**
 * @brief 检测结果结构体
 */
struct Detection {
    cv::Rect2f bbox;        // 边界框 (x, y, width, height)
    float confidence;       // 置信度
    int class_id;          // 类别ID
    std::string class_name; // 类别名称
};

/**
 * @brief YOLOv8圆形检测器类
 */
class YOLOv8Detector {
public:
    /**
     * @brief 构造函数
     * @param model_path ONNX模型路径
     * @param img_size 输入图像尺寸 (width, height)
     * @param obj_thresh 目标阈值
     * @param nms_thresh NMS阈值
     */
    YOLOv8Detector(const std::string& model_path, 
                   const cv::Size& img_size = cv::Size(640, 640),
                   float obj_thresh = 0.25f, 
                   float nms_thresh = 0.45f);

    /**
     * @brief 析构函数
     */
    ~YOLOv8Detector();

    /**
     * @brief 检测图像中的圆形对象
     * @param image 输入图像
     * @return 检测结果向量
     */
    std::vector<Detection> detect(const cv::Mat& image);

    /**
     * @brief 在图像上绘制检测结果
     * @param image 输入图像
     * @param detections 检测结果
     * @param draw_labels 是否绘制标签
     */
    void drawDetections(cv::Mat& image, 
                       const std::vector<Detection>& detections,
                       bool draw_labels = true);

    /**
     * @brief 设置置信度阈值
     * @param thresh 置信度阈值
     */
    void setObjectThreshold(float thresh) { obj_thresh_ = thresh; }

    /**
     * @brief 设置NMS阈值
     * @param thresh NMS阈值
     */
    void setNMSThreshold(float thresh) { nms_thresh_ = thresh; }

private:
    // ONNX Runtime相关
#ifdef HAVE_ONNXRUNTIME
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
#endif

    // 模型参数
    cv::Size img_size_;        // 输入图像尺寸
    float obj_thresh_;         // 目标阈值
    float nms_thresh_;         // NMS阈值

    // 类别信息
    static const std::vector<std::string> class_names_;
    static const std::vector<cv::Scalar> class_colors_;

    /**
     * @brief 初始化ONNX模型
     * @param model_path 模型路径
     */
    void initModel(const std::string& model_path);

    /**
     * @brief 图像预处理（letterbox）
     * @param image 输入图像
     * @param new_shape 目标尺寸
     * @param color 填充颜色
     * @param auto_pad 是否自动填充
     * @param scale_fill 是否拉伸填充
     * @param scale_up 是否允许放大
     * @param stride 步长
     * @return 处理后的图像和变换参数
     */
    std::tuple<cv::Mat, float, cv::Point2f> letterbox(
        const cv::Mat& image,
        const cv::Size& new_shape = cv::Size(640, 640),
        const cv::Scalar& color = cv::Scalar(114, 114, 114),
        bool auto_pad = true,
        bool scale_fill = false,
        bool scale_up = true,
        int stride = 32);

    /**
     * @brief 分布焦点损失(DFL)处理
     * @param position 位置预测张量
     * @return 处理后的张量
     */
    cv::Mat dfl(const cv::Mat& position);

    /**
     * @brief 边界框处理
     * @param position 位置预测
     * @return 处理后的边界框
     */
    cv::Mat boxProcess(const cv::Mat& position);

    /**
     * @brief 后处理ONNX模型输出
     * @param outputs 模型输出
     * @return 检测结果(boxes, classes, scores)
     */
    std::tuple<cv::Mat, cv::Mat, cv::Mat> postProcess(
        const std::vector<cv::Mat>& outputs);

    /**
     * @brief 过滤边界框
     * @param boxes 边界框
     * @param box_confidences 边界框置信度
     * @param box_class_probs 类别概率
     * @return 过滤后的结果
     */
    std::tuple<cv::Mat, cv::Mat, cv::Mat> filterBoxes(
        const cv::Mat& boxes,
        const cv::Mat& box_confidences,
        const cv::Mat& box_class_probs);

    /**
     * @brief 非极大值抑制
     * @param boxes 边界框
     * @param scores 分数
     * @return 保留的索引
     */
    std::vector<int> nmsBoxes(const cv::Mat& boxes, const cv::Mat& scores);

    /**
     * @brief 缩放边界框到原始图像尺寸
     * @param img1_shape 模型输入尺寸
     * @param boxes 边界框
     * @param img0_shape 原始图像尺寸
     * @param ratio_pad 缩放和填充参数
     * @return 缩放后的边界框
     */
    cv::Mat scaleBoxes(const cv::Size& img1_shape,
                       const cv::Mat& boxes,
                       const cv::Size& img0_shape,
                       const std::pair<float, cv::Point2f>& ratio_pad);

    /**
     * @brief 裁剪边界框到图像范围内
     * @param boxes 边界框
     * @param shape 图像尺寸
     */
    void clipBoxes(cv::Mat& boxes, const cv::Size& shape);

    /**
     * @brief 展平张量
     * @param input 输入张量
     * @return 展平后的张量
     */
    cv::Mat spFlatten(const cv::Mat& input);
};

} // namespace yolov8

#endif // YOLOV8_DETECTOR_H
