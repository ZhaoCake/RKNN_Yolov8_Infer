# RK_MODEL YOLOv8 6类圆形检测 C++ 实现

基于 C++ 和 ONNX Runtime 的 YOLOv8 6类圆形检测推理项目，使用 CMake 进行构建管理。

## 项目特点

针对沟槽的瑞芯微NPU，对YOLOv8模型有些自己的优化，还是自己写一遍才弄得清楚，单纯看一遍例程弄不清楚。

现在仅支持了转出的ONNX模型，后面再加RKNN。还没转换。

## 参考项目

[瑞芯微的yolov8 fork](https://github.com/airockchip/ultralytics_yolov8)
[RKNN模型库](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8)

## 系统要求

- C++17 兼容编译器 (GCC 7+, Clang 5+)
- CMake 3.16+
- OpenCV 4.x
- ONNX Runtime 1.16+ (推理需要)

## 快速开始

### 1. 安装依赖

#### Ubuntu/Debian
```bash
# 安装基础工具
sudo apt update
sudo apt install build-essential cmake pkg-config

# 安装OpenCV
sudo apt install libopencv-dev

# 安装ONNX Runtime 
./install_onnxruntime.sh
```

#### Arch Linux
```bash
# 安装基础工具
sudo pacman -S base-devel cmake pkg-config

# 安装OpenCV
sudo pacman -S opencv

# 安装ONNX Runtime
chmod +x install_onnxruntime.sh
./install_onnxruntime.sh
```

### 2. 构建项目

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 3. 运行推理

```bash
# 单张图像推理
./build/YOLOv8CircleDetection --model_path ../model/model.onnx --img_path ./images/test.jpg --img_show

# 批量推理并保存结果
./build/YOLOv8CircleDetection --model_path ../model/model.onnx --img_folder ./images --img_save

# 查看帮助
./build/YOLOv8CircleDetection --help
```

## 命令行参数

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| `--model_path` | string | ✓ | - | ONNX模型文件路径 |
| `--img_path` | string | ✗ | - | 单张图像路径 |
| `--img_folder` | string | ✗ | ./images | 图像文件夹路径 |
| `--img_show` | flag | ✗ | false | 显示检测结果 |
| `--img_save` | flag | ✗ | false | 保存结果到./result |
| `--obj_thresh` | float | ✗ | 0.25 | 目标检测阈值 |
| `--nms_thresh` | float | ✗ | 0.45 | NMS阈值 |
| `--help, -h` | flag | ✗ | false | 显示帮助信息 |

## 项目结构

```
YOLOv8_Inference/
├── CMakeLists.txt         # CMake构建配置
├── install_onnxruntime.sh # ONNX Runtime安装脚本
├── include/
│   └── yolov8_detector.h  # 检测器头文件
├── src/
│   ├── main.cc            # 主程序入口
│   └── yolov8_detector.cc # 检测器实现
├── reference/             # Python参考实现
├── images/                # 测试图像目录
├── result/                # 输出结果目录（自动创建）
├── model/                 # 模型文件目录
└── README.md              # 项目说明
```

## 检测类别

当前模型支持6种圆形对象检测（不提供数据集，自用）：

1. `blue_stack` - 蓝色堆 (红色框)
2. `blue_space` - 蓝色空 (青色框)  
3. `green_stack` - 绿色堆 (绿色框)
4. `green_space` - 绿色空 (黄色框)
5. `red_stack` - 红色堆 (蓝色框)
6. `red_space` - 红色空 (品红框)
