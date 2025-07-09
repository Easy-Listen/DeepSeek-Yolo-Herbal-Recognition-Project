import os
import cv2
import sys
import argparse
from datetime import datetime
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))
# 在文件顶部添加中文字体路径（根据你的系统选择）
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 思源黑体

from py_utils.coco_utils import COCO_test_helper


OBJ_THRESH = 0.25
NMS_THRESH = 0.45

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = (
    "白茯苓",    # 原"bai fu ling"
    "白芍",      # 原"bai shao"
    "白术",      # 原"bai zhu"
    "蒲公英",    # 原"pu gong ying"
    "甘草",      # 原"gan cao"
    "栀子",      # 原"zhi zi"
    "党参",      # 原"dang shen"
    "桃仁",      # 原"tao ren"
    "去皮桃仁",  # 原"qu pi tao ren"
    "地肤子",    # 原"di fu zi"
    "牡丹皮",    # 原"mu dan pi"
    "冬虫夏草",  # 原"dong chong xia cao"
    "杜仲",      # 原"du zhong"
    "当归",      # 原"dang gui"
    "杏仁",      # 原"xing ren"
    "何首乌",    # 原"he shou wu"
    "黄精",      # 原"huang jing"
    "鸡血藤",    # 原"ji xue teng"
    "枸杞",      # 原"gou qi"
    "莲须",      # 原"lian xu"
    "莲肉",      # 原"lian rou"
    "麦门冬",    # 原"mai men dong"
    "木通",      # 原"mu tong"
    "玉竹",      # 原"yu zhu"
    "女贞子",    # 原"nv zhen zi"
    "肉苁蓉",    # 原"rou cong rong"
    "人参",      # 原"ren shen"
    "乌梅",      # 原"wu mei"
    "覆盆子",    # 原"fu pen zi"
    "瓜蒌皮",    # 原"gua lou pi"
    "肉桂",      # 原"rou gui"
    "山茱萸",    # 原"shan zhu yu"
    "山药",      # 原"shan yao"
    "酸枣仁",    # 原"suan zao ren"
    "桑白皮",    # 原"sang bai pi"
    "山楂",      # 原"shan zha"
    "天麻",      # 原"tian ma"
    "熟地黄",    # 原"shu di huang"
    "小茴香",    # 原"xiao hui xiang"
    "泽泻",      # 原"ze xie"
    "竹茹",      # 原"zhu ru"
    "川贝母",    # 原"chuan bei mu"
    "川芎",      # 原"chuan xiong"
    "玄参",      # 原"xuan shen"
    "益智仁"     # 原"yi zhi ren"
)


coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

class TCMDetectionApp:
    def __init__(self, root, model, platform, args):
        self.root = root
        self.model = model
        self.platform = platform
        self.args = args
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        
        # 初始化状态
        self.cap = None
        self.is_camera_running = False
        self.current_frame = None
        self.detection_results = []
        self.last_update_time = 0
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 设置主窗口
        self.root.title("中药材智能识别系统")
        self.root.geometry("1200x800")
        
        # 创建界面组件
        self.create_widgets()
        
    def create_widgets(self):
        """创建所有GUI组件"""
        # 主框架布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 视频显示区域
        self.video_frame = ttk.LabelFrame(main_frame, text="实时画面")
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0,10))
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # 控制按钮区域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.camera_btn = ttk.Button(control_frame, text="开启摄像头", command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        self.open_btn = ttk.Button(control_frame, text="打开图片", command=self.open_image)
        self.open_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_btn = ttk.Button(control_frame, text="拍照", command=self.capture_image, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        self.detect_btn = ttk.Button(control_frame, text="检测", command=self.detect_image, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(control_frame, text="保存结果", command=self.save_result, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="检测结果")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(result_frame, height=10)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(self.result_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar.config(command=self.result_text.yview)
        self.result_text.config(yscrollcommand=scrollbar.set)
        
        # 状态栏
        self.status_bar = ttk.Label(main_frame, text="准备就绪", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(5,0))
        
        # FPS显示
        self.fps_label = ttk.Label(main_frame, text="FPS: 0")
        self.fps_label.pack(anchor=tk.E)
    
    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """启动摄像头并设置画面尺寸与显示框一致"""
        try:
            # 初始化摄像头
            self.cap = cv2.VideoCapture("/dev/video11")
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")
            
            # 获取显示框的尺寸
            display_width = self.video_label.winfo_width()
            display_height = self.video_label.winfo_height()
        
            # 如果显示框还没有尺寸（首次启动），使用默认尺寸
            if display_width <= 1 or display_height <= 1:
                display_width = 640
                display_height = 480
            
            # 设置摄像头分辨率（尽量匹配显示框尺寸）
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
        
            # 强制设置摄像头分辨率（某些摄像头可能需要）
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
            self.is_camera_running = True
            self.camera_btn.config(text="关闭摄像头")
            self.capture_btn.config(state=tk.NORMAL)
            self.detect_btn.config(state=tk.NORMAL)
            self.status_bar.config(text="摄像头已开启")
        
            # 开始更新画面
            self.update_frame()
        except Exception as e:
            messagebox.showerror("错误", f"无法启动摄像头: {str(e)}")
    
    def stop_camera(self):
        """停止摄像头"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_camera_running = False
        self.camera_btn.config(text="开启摄像头")
        self.capture_btn.config(state=tk.DISABLED)
        self.detect_btn.config(state=tk.DISABLED)
        self.status_bar.config(text="摄像头已停止")
    
    def update_frame(self):
        """更新摄像头画面，保持与显示框尺寸一致"""
        if self.is_camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # 计算FPS
                current_time = time.time()
                self.frame_count += 1
            
                if current_time - self.start_time >= 1:
                    self.fps = self.frame_count / (current_time - self.start_time)
                    self.fps_label.config(text=f"FPS: {self.fps:.1f}")
                    self.start_time = current_time
                    self.frame_count = 0
            
                # 存储当前帧
                self.current_frame = frame.copy()
            
                # 获取显示框的当前尺寸
                display_width = self.video_label.winfo_width()
                display_height = self.video_label.winfo_height()
            
                # 如果显示框还没有尺寸（首次启动），使用默认尺寸
                if display_width <= 1 or display_height <= 1:
                    display_width = 640
                    display_height = 480
            
                # 转换为Tkinter可显示的格式
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(image=img)
            
                # 更新显示
                self.video_label.img = img
                self.video_label.config(image=img)
        
            # 继续更新
            self.root.after(30, self.update_frame)
    
    def open_image(self):
        """打开图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    self.current_frame = img
                    
                    # 显示图片
                    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_display = Image.fromarray(img_display)
                    img_display = ImageTk.PhotoImage(image=img_display)
                    
                    self.video_label.img = img_display
                    self.video_label.config(image=img_display)
                    
                    self.detect_btn.config(state=tk.NORMAL)
                    self.save_btn.config(state=tk.NORMAL)
                    self.status_bar.config(text=f"已加载: {os.path.basename(file_path)}")
                else:
                    raise Exception("无法读取图片")
            except Exception as e:
                messagebox.showerror("错误", f"无法打开图片: {str(e)}")
    
    def detect_image(self):
        """检测当前显示的图像"""
        if self.current_frame is not None:
            try:
                # 处理图像
                processed_frame = self.process_frame(self.current_frame.copy())
                
                # 显示处理后的图像
                img_display = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img_display = Image.fromarray(img_display)
                img_display = ImageTk.PhotoImage(image=img_display)
                
                self.video_label.img = img_display
                self.video_label.config(image=img_display)
                
                # 更新结果文本
                self.update_result_text()
                
                self.status_bar.config(text="检测完成")
            except Exception as e:
                messagebox.showerror("错误", f"检测失败: {str(e)}")
    
    def process_frame(self, frame):
        """处理帧图像"""
        # 预处理
        img = self.co_helper.letter_box(
            im=frame.copy(), 
            new_shape=(IMG_SIZE[1], IMG_SIZE[0]), 
            pad_color=(0,0,0)
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 根据平台类型准备输入数据
        if self.platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2, 0, 1))
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32) / 255.0
        else:
            input_data = img
        
        # 运行模型
        outputs = self.model.run([input_data])
        boxes, classes, scores = post_process(outputs)
        
        # 存储检测结果
        self.detection_results = []
        if boxes is not None:
            real_boxes = self.co_helper.get_real_box(boxes)
            for box, score, cl in zip(real_boxes, scores, classes):
                self.detection_results.append({
                    'class': CLASSES[cl],
                    'score': score,
                    'box': box
                })
            
            # 绘制检测结果
            draw(frame, real_boxes, scores, classes)
        
        return frame
    
    def update_result_text(self):
        """更新结果文本框"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        if self.detection_results:
            for result in self.detection_results:
                line = f"{result['class']}: 置信度 {result['score']:.2f} @ {result['box']}\n"
                self.result_text.insert(tk.END, line)
        else:
            self.result_text.insert(tk.END, "未检测到中药材\n")
        
        self.result_text.config(state=tk.DISABLED)
    
    def capture_image(self):
        """拍照保存当前帧"""
        if self.current_frame is not None:
            try:
                # 创建保存目录
                os.makedirs("captures", exist_ok=True)
                
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captures/capture_{timestamp}.jpg"
                
                # 保存图片
                cv2.imwrite(filename, self.current_frame)
                
                # 更新状态
                self.status_bar.config(text=f"照片已保存: {filename}")
                messagebox.showinfo("成功", f"照片已保存到: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def save_result(self):
        """保存检测结果"""
        if self.current_frame is not None and self.detection_results:
            file_path = filedialog.asksaveasfilename(
                title="保存结果",
                defaultextension=".jpg",
                filetypes=[("JPEG 图片", "*.jpg"), ("PNG 图片", "*.png")]
            )
            
            if file_path:
                try:
                    cv2.imwrite(file_path, self.current_frame)
                    
                    # 同时保存文本结果
                    txt_path = os.path.splitext(file_path)[0] + ".txt"
                    with open(txt_path, "w") as f:
                        for result in self.detection_results:
                            f.write(f"{result['class']}: {result['score']:.3f} @ {result['box']}\n")
                    
                    self.status_bar.config(text=f"结果已保存到: {file_path}")
                    messagebox.showinfo("成功", f"结果已保存到:\n{file_path}\n{txt_path}")
                except Exception as e:
                    messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def on_closing(self):
        """关闭窗口时的清理工作"""
        self.stop_camera()
        if hasattr(self.model, 'release'):
            self.model.release()
        self.root.destroy()

def draw(image, boxes, scores, classes):
    """绘制检测结果（支持中文），标签显示在框内左上角"""
    # 转换OpenCV图像为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # 尝试加载中文字体，适当减小字体大小以便框内显示
        font = ImageFont.truetype(FONT_PATH, 18, encoding="utf-8")  # 从20减小到18
    except:
        # 回退到默认字体（可能不支持中文）
        font = ImageFont.load_default()
        print("警告：无法加载中文字体，将使用默认字体")
    
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        
        # 绘制矩形框
        draw.rectangle([top, left, right, bottom], outline=(255, 0, 0), width=2)
        
        # 绘制中文文本
        text = f"{CLASSES[cl]} {score:.2f}"
        
        # 获取文本尺寸（兼容新旧Pillow版本）
        try:
            # 新版本Pillow使用textbbox
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # 旧版本Pillow使用textsize
            text_width, text_height = draw.textsize(text, font=font)
        
        # 计算文本位置（框内左上角，向右下方偏移5像素）
        text_x = top + 5  # 向右偏移5像素
        text_y = left + 5  # 向下偏移5像素
        
        # 绘制半透明文本背景（增强可读性）
        bg_color = (255, 0, 0, 128)  # 红色半透明背景
        draw.rectangle(
            [text_x, text_y, text_x + text_width, text_y + text_height],
            fill=bg_color
        )
        
        # 绘制文本（白色文字更醒目）
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    # 转换回OpenCV格式
    image[:] = cv2.cvtColor(np.array(img_pil, dtype=np.uint8), cv2.COLOR_RGB2BGR)

def box_process(position):
    """处理模型输出的边界框坐标"""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)
    
    position = position.transpose(0,2,3,1)
    box_xy = grid + position[..., 0:2]
    box_xy = box_xy * stride / IMG_SIZE
    box_wh = position[..., 2:4] / IMG_SIZE
    box_xywh = np.concatenate((box_xy, box_wh), axis=-1)
    
    # 转换为xyxy格式
    box_xyxy = np.zeros_like(box_xywh)
    box_xyxy[..., 0] = box_xywh[..., 0] - box_xywh[..., 2] / 2  # x1
    box_xyxy[..., 1] = box_xywh[..., 1] - box_xywh[..., 3] / 2  # y1
    box_xyxy[..., 2] = box_xywh[..., 0] + box_xywh[..., 2] / 2  # x2
    box_xyxy[..., 3] = box_xywh[..., 1] + box_xywh[..., 3] / 2  # y2
    
    return box_xyxy
    
def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # 根据阈值过滤
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # NMS处理
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def filter_boxes(boxes, box_confidences, box_class_probs):
    """根据对象阈值过滤框"""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """抑制非极大值框"""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--target', type=str, default='rk3566')
    parser.add_argument('--device_id', type=str, default=None)
    args = parser.parse_args()

    # 初始化模型
    model, platform = setup_model(args)
    
    # 创建GUI应用
    root = tk.Tk()
    app = TCMDetectionApp(root, model, platform, args)
    
    # 设置关闭事件处理
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # 启动主循环
    root.mainloop()

if __name__ == '__main__':
    import time
    main()
