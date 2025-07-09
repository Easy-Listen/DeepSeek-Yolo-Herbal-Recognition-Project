# 基于RK3588芯片的嵌入式AI药材识别系统

## 📁 项目目录结构

### 1. RKLLM大语言模型模块

    rkllm/
    ├── lib/                  # 核心推理库
    │   └── librkllmrt.so     # Deepseek-R1 NPU加速库
    ├── build-linux.sh        # 交叉编译脚本
    ├── CMakeLists.txt        # 跨平台构建配置
    ├── convert.py            # 模型转换工具
    ├── llm_demo_6            # 可执行文件
    └── llm_demo_6.cpp       # 交互终端源码

### 2. YOLOv11视觉识别模块

    yolo11/
    ├── cpp/                  # C++推理实现 
    ├── datasets/             # 训练数据集
    ├── model/                # 模型文件
        ├── best.onnx                  # 原始ONNX模型  
        ├── herb.rknn                  # 主RKNN模型 
        ├── coco_45_labels_list.txt    # 类别标签 
        ├── dataset.txt                # 数据集清单
        └── download_model.sh          # 模型下载脚本
    ├── python/               # Python训练工具
        ├── convert.py                # 将ONNX模型转为RKNN格式
        └── yolo11_17.py          # 最终项目运行脚本
    ├── test_images/          # 测试图像库 
    ├── best.pt               # 最优权重 
    ├── yolo11.yaml           # 模型配置 
    ├── train.py              # 训练脚本 
    └── data.yaml             # 数据配置 
### 3. 技术文档
    技术文档.pdf
### 4. 演示视频
    演示视频.mp4
## 🔧 核心组件说明

### 一、RKLLM模块
| 组件 | 功能说明 |
|----------------------|--------------------------------------------------------------------------|
| `lib/librkllmrt.so` | 大语言模型Deepseek-R1推理的核心动态链接库，属于RKLLM工具链组件          |
| `build-linux.sh`    | 交叉编译构建脚本：<br>• 使用GCC工具链为ARM架构编译<br>• 支持多线程(-j4)编译<br>• 可指定Release/Debug模式 |
| `CMakeLists.txt`    | CMake构建配置：<br>• 编译llm_demo_6和multimodel_demo可执行文件<br>• 支持Linux/Android跨平台链接<br>• 依赖librkllmrt.so库 |
| `convert.py`        | 模型转换工具：<br>• 将HuggingFace格式模型转为RK3588专用格式<br>• 支持w8a8量化<br>• 输出NPU加速的.rkllm文件 |
| `llm_demo_6.cpp`    | 大语言模型交互终端：<br>• 加载量化后的DeepSeek-R1模型<br>• 支持200字内中文问答<br>• 实时流式输出<br>• 安全退出处理 |
|
### 二、YOLOv11模块
| 组件 | 功能说明 |
|----------------------|----------------------------------------------------------|
| `python/` | Python脚本            |
| `datasets/`    | COCO格式训练数据      |
| `test_images/` | 测试用样本图像 |
|

| 文件 | 类型  |功能描述 |
|------|------|----------|
| `best.pt` | PyTorch模型 |预训练权重文件，支持45类中药材识别 |
| `yolo11.yaml` | 配置文件 |  模型架构定义文件 |
| `data.yaml` | 配置文件 |  数据集路径和类别配置 |
| `train.py` | Python脚本 | 模型训练主程序 |
| `python/convert.py` | Python脚本 | 模型格式转换工具：<br>• ONNX→RKNN转换<br>• 支持W8A8量化<br>• 输出NPU优化模型 |
| `python/yolo11_17.py` | Python脚本 | 主运行程序：<br>• 加载RKNN模型<br>• 实时图像推理<br>• 结果可视化输出 |
|
## 🔍 核心模型说明

| 文件 | 类型 | 大小 | 用途 |
|------|------|------|------|
| `best.onnx` | ONNX模型 | 10.2MB | YOLOv11原始导出模型 |
| `herb.rknn` | RKNN模型 | 4.1MB | 量化后的主推理模型 |
| `coco_45_labels_list.txt` | 文本 | 1KB | 45类中药材标签 |
|
## ⚠️ 注意事项
1. RKNN模型需在RK3588 NPU环境运行
2. ONNX文件可用于PC端验证
## 📌 版本信息
- **最后更新**：2025-07-10
- **依赖环境**：
  - PyTorch ≥1.8
  - RKNN-Toolkit2 v1.6+
  - Ubuntu 22.04 LTS