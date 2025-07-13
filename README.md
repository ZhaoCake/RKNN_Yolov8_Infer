# YOLOv8 ONNX C++ 推理项目

这是一个使用C++和ONNX Runtime对YOLOv8模型进行推理的项目，专门用于检测circle数据集中的6个类别。

## 项目结构

```
YOLOv8_Inference/
├── xmake.lua              # xmake构建配置文件
├── dataset.yaml           # 数据集配置文件
├── include/
│   └── yolov8_detector.h  # YOLOv8检测器头文件
├── src/
│   ├── yolov8_detector.cpp # YOLOv8检测器实现
│   └── main.cpp           # 主程序
├── demo/
│   └── main.cpp           # 演示程序
├── reference/             # Python参考实现
│   ├── yolov8_simple.py   # 简化版Python推理 (推荐)
│   ├── yolov8_circle.py   # 完整版Python推理
│   ├── analyze_model.py   # 模型分析工具
│   └── README.md          # Python版本使用说明
├── requirements.txt       # Python依赖
└── README.md              # 说明文档
```

## 依赖库

- **OpenCV**: 用于图像处理
- **ONNX Runtime**: 用于ONNX模型推理
- **xmake**: 项目构建工具

## 安装依赖

### 1. 安装xmake

```bash
# Ubuntu/Debian
curl -fsSL https://xmake.io/shget.text | bash

# 或者通过包管理器
sudo add-apt-repository ppa:xmake-io/xmake
sudo apt update
sudo apt install xmake
```

### 2. 安装系统依赖

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake git

# 安装OpenCV
sudo apt install libopencv-dev

# 安装ONNX Runtime (或者通过xmake自动安装)
# 下载预编译版本: https://github.com/microsoft/onnxruntime/releases
```

## 构建项目

```bash
# 进入项目目录
cd /home/zhaocake/WorkSpace/Vision/YOLOv8_Inference

# 配置项目 (第一次运行会自动下载依赖)
xmake config

# 编译项目
xmake build

# 或者直接运行 (会自动编译)
xmake run yolov8_inference <model_path> <image_path>
```

## 使用方法

### Python版本 (推荐用于快速测试)

首先安装Python依赖：
```bash
pip install -r requirements.txt
```

#### 1. 分析模型输出格式
```bash
python reference/analyze_model.py --model your_model.onnx
```

#### 2. 运行推理
```bash
# 单张图像
python reference/yolov8_simple.py --model your_model.onnx --source test_image.jpg --show --save

# 批量处理
python reference/yolov8_simple.py --model your_model.onnx --source images_folder/ --save

# 调整参数
python reference/yolov8_simple.py --model your_model.onnx --source test.jpg --conf 0.3 --iou 0.5
```

### C++版本 (用于生产部署)

#### 1. 准备ONNX模型

确保你已经将训练好的YOLOv8模型导出为ONNX格式：

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('path/to/your/best.pt')

# 导出为ONNX格式
model.export(format='onnx', imgsz=640)
```

#### 2. 构建C++项目
```bash
# 进入项目目录
cd /home/zhaocake/WorkSpace/Vision/YOLOv8_Inference

# 配置项目 (第一次运行会自动下载依赖)
xmake config

# 编译项目
xmake build
```

#### 3. 运行C++推理

```bash
# 基本用法
xmake run yolov8_inference model.onnx test_image.jpg

# 或者运行演示程序
xmake run demo model.onnx test_image.jpg
```

### 开发建议

1. **快速原型**: 使用Python版本进行快速测试和调试
2. **模型验证**: 使用`analyze_model.py`分析模型输出格式
3. **生产部署**: 使用C++版本获得最佳性能
4. **参数调优**: 在Python版本中找到最佳参数后，移植到C++版本

### 3. 输出结果

程序会在控制台输出检测结果，并保存可视化结果到 `detection_result.jpg`。

输出示例：
```
Loading YOLOv8 model...
Model loaded successfully!
Input shape: [1, 3, 640, 640]
Output shape: [1, 84, 8400]
Image loaded: 1920x1080
Running inference...
Found 3 objects:
  1. blue_stack (confidence: 85.67%) bbox: [245, 123, 68, 72]
  2. red_space (confidence: 92.34%) bbox: [456, 234, 89, 91]
  3. green_stack (confidence: 78.45%) bbox: [678, 345, 76, 82]
Result saved to: detection_result.jpg
```

## 类别说明

根据数据集配置，模型可以检测以下6个类别：

- `blue_stack` (ID: 0)
- `blue_space` (ID: 1) 
- `green_stack` (ID: 2)
- `green_space` (ID: 3)
- `red_stack` (ID: 4)
- `red_space` (ID: 5)

## 配置参数

可以在 `yolov8_detector.cpp` 中修改以下参数：

- `conf_threshold_`: 置信度阈值 (默认: 0.25)
- `iou_threshold_`: NMS IoU阈值 (默认: 0.45)
- `input_size_`: 输入图像尺寸 (默认: 640x640)

## 性能优化

1. **使用GPU推理**: 如果有NVIDIA GPU，可以安装ONNX Runtime GPU版本
2. **模型量化**: 可以使用INT8量化模型提升推理速度
3. **批处理**: 对于多张图像，可以使用批处理推理

## 故障排除

### 常见问题

1. **依赖库找不到**: 确保OpenCV和ONNX Runtime正确安装
2. **模型加载失败**: 检查ONNX模型路径和格式
3. **编译错误**: 确保C++17支持和所有依赖库版本兼容

### Debug模式

```bash
# 使用debug模式编译
xmake config -m debug
xmake build

# 运行debug版本
xmake run yolov8_inference model.onnx image.jpg
```

## 许可证

本项目仅供学习和研究使用。
