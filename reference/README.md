# YOLOv8 Circle Detection Python 推理

这个目录包含了针对修改后的YOLOv8 ONNX模型的Python推理代码，专门用于检测circle数据集中的6个类别。该代码适配RKNN导出的特殊ONNX模型结构（移除后处理、DFL结构外移、置信度分支）。

## 文件说明

- `yolov8_circle.py`: 标准推理实现，包含完整的DFL后处理、NMS和坐标映射逻辑
- `analyze_model.py`: 模型结构分析工具，用于查看模型输入输出维度
- `README.md`: 使用说明文档

## 安装依赖

```bash
# 安装Python依赖（在项目根目录）
pip install -r requirements.txt

# 或者手动安装
pip install opencv-python onnxruntime numpy
```

## 使用方法

### 1. 单张图像推理

```bash
python reference/yolov8_circle.py --model_path model/circle_yolov8n_250713.onnx --img_path images/your_image.jpg --img_save --img_show
```

### 2. 批量图像推理

```bash
python reference/yolov8_circle.py --model_path model/circle_yolov8n_250713.onnx --img_folder images --img_save
```

### 3. 仅推理不保存结果

```bash
python reference/yolov8_circle.py --model_path model/circle_yolov8n_250713.onnx --img_folder images
```

## 参数说明

- `--model_path`: ONNX模型路径 (必需)
- `--img_path`: 单张图像路径 (可选，与img_folder二选一)
- `--img_folder`: 图像文件夹路径 (默认: ./test_images)
- `--img_save`: 保存检测结果到result文件夹
- `--img_show`: 显示检测结果窗口

## 检测阈值调整

可以在代码中修改以下参数：

```python
OBJ_THRESH = 0.25    # 置信度阈值
NMS_THRESH = 0.45    # NMS IoU阈值
```

## 模型结构适配

代码适配了RKNN导出的修改后YOLOv8模型结构：

1. **DFL后处理外移**: Distribution Focal Loss计算在CPU端实现
2. **三分支输出**: 每个检测尺度包含DFL、类别、置信度三个分支
3. **置信度分离**: 支持独立的置信度分数分支
4. **坐标映射优化**: 修正了letterbox预处理的坐标变换逻辑，确保大分辨率图像检测框准确映射

## 核心功能

### 后处理流程
1. **DFL处理**: 对边界框分布进行softmax和加权求和
2. **网格生成**: 为每个检测尺度生成对应的网格坐标
3. **坐标解码**: 将网格相对坐标转换为绝对坐标
4. **置信度筛选**: 基于目标置信度和类别置信度进行筛选
5. **NMS**: 对每个类别分别进行非极大值抑制
6. **坐标映射**: 将检测框从模型输入尺寸映射回原始图像尺寸

## 检测类别

根据dataset.yaml配置，支持以下6个类别：

- **blue_stack** (ID: 0) - 红色框
- **blue_space** (ID: 1) - 青色框
- **green_stack** (ID: 2) - 绿色框
- **green_space** (ID: 3) - 黄色框
- **red_stack** (ID: 4) - 蓝色框
- **red_space** (ID: 5) - 紫色框

## 输出示例

```bash
Model-model/circle_yolov8n_250713.onnx is onnx model, starting inference
infer 1/1

IMG: test_image.jpg
blue_stack @ (245 123 313 195) 0.857
red_space @ (456 234 545 325) 0.923
green_stack @ (678 345 754 427) 0.784
Detection result save to ./result/test_image.jpg
Inference completed!
```

## 注意事项

1. **模型兼容性**: 代码专门针对RKNN优化的YOLOv8模型设计，适配了特殊的输出格式
2. **后处理精度**: DFL和NMS等操作在CPU上执行，确保结果准确性
3. **坐标映射**: 已修正大分辨率图像的坐标映射问题，确保检测框正确对应原图
4. **输出目录**: 检测结果保存在`./result/`目录下
5. **支持格式**: 支持jpg, jpeg, png, bmp格式的图像

## 性能优化

- 针对RKNN模型结构进行了专门优化
- 支持批量图像处理
- 内存使用优化，适合大量图像推理
- 高效的后处理流程，减少不必要的计算开销

## 调试说明

## 调试说明

如果检测结果不正确，可以：

1. **分析模型结构**: 使用`analyze_model.py`查看模型输入输出维度
   ```bash
   python reference/analyze_model.py model/circle_yolov8n_250713.onnx
   ```

2. **调整检测阈值**: 修改代码中的`OBJ_THRESH`和`NMS_THRESH`参数
3. **验证类别数量**: 确认模型输出对应6个类别
4. **检查图像预处理**: 确认letterbox操作与训练时一致
5. **坐标映射验证**: 如果检测框位置不准确，检查`scale_boxes`函数的参数传递

## 常见问题

**Q: 检测框位置偏移怎么办？**
A: 代码已修正坐标映射逻辑，确保letterbox预处理的padding和缩放参数正确传递给后处理函数。

**Q: 没有检测到任何目标怎么办？**
A: 尝试降低`OBJ_THRESH`阈值，或者检查模型输出格式是否匹配代码中的假设。

**Q: 如何验证模型是否正确加载？**
A: 运行时会输出模型路径和平台信息，确认模型成功加载。

## 扩展说明

本推理代码可以作为基础框架，适配其他类似的RKNN导出YOLOv8模型。主要需要调整的部分：
- 类别数量和名称（`CLASSES`变量）
- 检测阈值（`OBJ_THRESH`, `NMS_THRESH`）
- 输入尺寸（`IMG_SIZE`）
- 颜色映射（`CLASS_COLORS`）
