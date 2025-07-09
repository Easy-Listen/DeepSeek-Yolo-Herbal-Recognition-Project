import os
import cv2
import sys
import csv  
import argparse
import time
import datetime
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import subprocess

# 添加路径
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

# 中文字体路径
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

from py_utils.coco_utils import COCO_test_helper

# 检测参数
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height)
# AI模型参数
RESPONSE_TERMINATOR = "<|endoftext|>"

# 中药材类别
CLASSES = (
    "白茯苓", "白芍", "白术", "蒲公英", "甘草", "栀子", "党参", "桃仁", "去皮桃仁", 
    "地肤子", "牡丹皮", "冬虫夏草", "杜仲", "当归", "杏仁", "何首乌", "黄精", 
    "鸡血藤", "枸杞", "莲须", "莲肉", "麦门冬", "木通", "玉竹", "女贞子", 
    "肉苁蓉", "人参", "乌梅", "覆盆子", "瓜蒌皮", "肉桂", "山茱萸", "山药", 
    "酸枣仁", "桑白皮", "山楂", "天麻", "熟地黄", "小茴香", "泽泻", "竹茹", 
    "川贝母", "川芎", "玄参", "益智仁"
)

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 
                21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                41, 42, 43, 44, 45]
class DeepseekRKLLM:
    def __init__(self):
        self.deepseek_rkllm_path = "/media/elf/EXT4/rkllm/deepseek-r1-1.5b-w8a8.rkllm"
        self.llm_demo_path = "/media/elf/EXT4/rkllm/llm_demo_5"
        self.lib_path = "/media/elf/EXT4/rkllm/lib"
        self.process = None
        self._init_process()

    def _validate_paths(self):
        if not os.path.exists(self.deepseek_rkllm_path):
            raise FileNotFoundError(f"模型文件不存在: {self.deepseek_rkllm_path}")
        if not os.path.exists(self.llm_demo_path):
            raise FileNotFoundError(f"可执行文件不存在: {self.llm_demo_path}")
        if not os.access(self.llm_demo_path, os.X_OK):
            raise PermissionError(f"可执行文件没有权限: {self.llm_demo_path}")

    def _init_process(self, max_new_tokens=256, num_beams=256):
        """初始化并启动子进程"""
        self._validate_paths()
        
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{self.lib_path}:" + env.get("LD_LIBRARY_PATH", "")
        
        cmd = [
            self.llm_demo_path,
            self.deepseek_rkllm_path,
            str(max_new_tokens),
            str(num_beams)
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # 等待初始化完成
        output = []
        while True:
            line = self.process.stdout.readline()
            if not line and self.process.poll() is not None:
                break
            if "rkllm init success" in line:
                output.append(line)
                break

        return "".join(output) if output else "无响应"

    def generate(self, prompt):
        """生成文本（单次交互）"""
        if not self.process:
            raise RuntimeError("子进程未启动")

        self.process.stdin.write(f"{prompt}\n")
        self.process.stdin.flush()

        output = []
        while True:
            line = self.process.stdout.readline()
            if not line:  # 管道关闭
                break
            line = line.strip()
            if line:
                output.append(line)
            if line.endswith(RESPONSE_TERMINATOR):
                output[-1] = line[:-len(RESPONSE_TERMINATOR)].strip()
                break
                
        return "\n".join(output)
    
    def __del__(self):
        """清理资源"""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
            except:
                pass

def dfl(position):
    """Distribution Focal Loss解码"""
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y.numpy()

def box_process(position):
    """处理YOLO输出的边界框坐标"""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:,0:2,:,:]
    box_xy2 = grid + 0.5 + position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

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

def post_process(input_data):
    """后处理模型输出"""
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
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

def draw(image, boxes, scores, classes):
    """绘制检测结果（支持中文）"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype(FONT_PATH, 18, encoding="utf-8")
    except:
        font = ImageFont.load_default()
        print("警告：无法加载中文字体，将使用默认字体")
    
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        
        # 绘制矩形框
        draw.rectangle([top, left, right, bottom], outline=(255, 0, 0), width=2)
        
        # 绘制中文文本
        text = f"{CLASSES[cl]} {score:.2f}"
        
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)
        
        text_x = top + 5
        text_y = left + 5
        
        # 绘制半透明背景
        bg_color = (255, 0, 0, 128)
        draw.rectangle(
            [text_x, text_y, text_x + text_width, text_y + text_height],
            fill=bg_color
        )
        
        # 绘制文本
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    image[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

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
        self.spreadsheet_path = "herb_records.csv" # 电子表单文件路径
        self.init_spreadsheet()
        
        # AI模型状态
        self.llm = None
        self.is_ai_initializing = False
        
        # 添加AI模型状态变量
        self.ai_model_active = False  # 跟踪AI模型是否处于活动状态
                
        # 设置主窗口
        self.root.title("中药材智能识别系统")
        self.root.geometry("1400x900")
        
        # 创建界面组件
        self.create_widgets()

    def init_spreadsheet(self):
        """初始化电子表单文件"""
        if not os.path.exists(self.spreadsheet_path):
            with open(self.spreadsheet_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['时间', '药材名称', '置信度', '位置(x1,y1,x2,y2)'])
                        
    def create_widgets(self):
        """创建所有GUI组件"""
        # 主框架布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧检测区域
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 右侧AI区域
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10,0))
        
        # ========== 左侧检测区域 ==========
        # 视频显示区域
        self.video_frame = ttk.LabelFrame(left_frame, text="实时画面")
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0,10))
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # 控制按钮区域
        control_frame = ttk.Frame(left_frame)
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

        self.export_btn = ttk.Button(control_frame, text="加入电子表单", 
                                   command=self.add_to_spreadsheet, 
                                   state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)   
             
        # 结果显示区域
        result_frame = ttk.LabelFrame(left_frame, text="检测结果")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(result_frame, height=10)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(self.result_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar.config(command=self.result_text.yview)
        self.result_text.config(yscrollcommand=scrollbar.set)
        
        # ========== 右侧AI区域 ==========
        # AI初始化区域
        ai_init_frame = ttk.LabelFrame(right_frame, text="AI模型初始化")
        ai_init_frame.pack(fill=tk.X, pady=(0,10))
        
        self.ai_init_btn = ttk.Button(
            ai_init_frame, 
            text="初始化AI模型", 
            command=self.toggle_ai_model,
            style='Accent.TButton'
        )
        self.ai_init_btn.pack(pady=5)
        
        self.ai_status_label = ttk.Label(
            ai_init_frame,
            text="AI模型未初始化",
            foreground="red"
        )
        self.ai_status_label.pack()
        
        # AI问答区域
        ai_qa_frame = ttk.LabelFrame(right_frame, text="中药材知识问答")
        ai_qa_frame.pack(fill=tk.BOTH, expand=True)
        
        self.ai_chat_text = scrolledtext.ScrolledText(
            ai_qa_frame,
            wrap=tk.WORD,
            height=15
        )
        self.ai_chat_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ai_input_frame = ttk.Frame(ai_qa_frame)
        ai_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.ai_input = ttk.Entry(ai_input_frame)
        self.ai_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ai_input.bind("<Return>", lambda e: self.ask_ai_question())
        
        self.ai_ask_btn = ttk.Button(
            ai_input_frame,
            text="提问",
            command=self.ask_ai_question,
            state=tk.DISABLED
        )
        self.ai_ask_btn.pack(side=tk.RIGHT, padx=(5,0))
        
        # 状态栏
        self.status_bar = ttk.Label(main_frame, text="准备就绪", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(5,0))
        
        # FPS显示
        self.fps_label = ttk.Label(main_frame, text="FPS: 0")
        self.fps_label.pack(anchor=tk.E)

    # ========== AI模型相关方法 ==========
    def toggle_ai_model(self):
        """切换AI模型状态（初始化/关闭）"""
        if not self.ai_model_active:
            self.start_ai_initialization()
        else:
            self.shutdown_ai_model()

    def shutdown_ai_model(self):
        """关闭AI模型（修正版）"""
        try:
            if self.llm:
                # 确保先关闭子进程再释放资源
                if hasattr(self.llm, 'process') and self.llm.process:
                    self.llm.process.terminate()
                del self.llm
            self.llm = None
            self.ai_model_active = False
        
            # 强制更新按钮状态（确保可操作）
            self.ai_init_btn.config(
                text="初始化AI模型",
                state=tk.NORMAL,  # 明确设置为可点击状态
                style='Accent.TButton'  # 确保使用正确的样式
            )
        
            # 更新其他UI状态
            self.ai_status_label.config(text="AI模型已关闭", foreground="red")
            self.ai_ask_btn.config(state=tk.DISABLED)
            self.append_ai_message("系统", "AI模型已安全关闭")
            self.status_bar.config(text="AI模型已成功关闭")
        
            # 调试输出（完成后可删除）
            print("[DEBUG] 按钮当前状态:", self.ai_init_btn["state"])
        except Exception as e:
            messagebox.showerror("错误", f"关闭模型时出错: {str(e)}")
            # 出错时也确保按钮可用
            self.ai_init_btn.config(state=tk.NORMAL)
            
    def start_ai_initialization(self):
        """开始初始化AI模型"""
        if self.is_ai_initializing:
            messagebox.showinfo("提示", "AI模型正在初始化中，请稍候...")
            return
            
        self.is_ai_initializing = True
        self.ai_init_btn.config(state=tk.DISABLED)
        self.ai_status_label.config(text="AI模型初始化中...", foreground="orange")
        self.status_bar.config(text="正在初始化AI模型，这可能需要几分钟时间...")
        
        # 在后台线程中初始化AI模型
        threading.Thread(
            target=self.initialize_ai_model,
            daemon=True
        ).start()
        
        # 定期检查初始化状态
        self.root.after(100, self.check_ai_initialization_status)
    
    def initialize_ai_model(self):
        """初始化AI模型"""
        try:
            self.llm = DeepseekRKLLM()
            self.is_ai_initializing = False
        except Exception as e:
            self.is_ai_initializing = False
            self.root.after(0, lambda: messagebox.showerror("AI初始化失败", str(e)))
    
    def check_ai_initialization_status(self):
        """线程安全的初始化状态检查"""
        if self.is_ai_initializing:
            self.root.after(100, self.check_ai_initialization_status)
        else:
            try:
                if self.llm is not None:
                    # 使用invoke确保在主线程更新UI
                    self.root.after(0, lambda: [
                        setattr(self, 'ai_model_active', True),
                        self.ai_init_btn.config(
                            text="关闭AI模型", 
                            state=tk.NORMAL  # 明确设置状态
                        ),
                        self.ai_status_label.config(text="AI模型运行中", foreground="green"),
                        self.ai_ask_btn.config(state=tk.NORMAL)
                    ])
                else:
                    self.root.after(0, lambda: [
                        self.ai_init_btn.config(
                            text="初始化AI模型",
                            state=tk.NORMAL  # 失败时也保持可点击
                        ),
                        self.ai_status_label.config(text="AI模型未初始化", foreground="red")
                    ])
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("状态更新错误", str(e)))
    
    def ask_ai_question(self):
        """向AI提问"""
        question = self.ai_input.get().strip()
        if not question:
            return
            
        self.append_ai_message("用户", question)
        self.ai_input.delete(0, tk.END)
        self.ai_ask_btn.config(state=tk.DISABLED)
        
        # 在后台线程中处理问题
        threading.Thread(
            target=self.process_ai_question,
            args=(question,),
            daemon=True
        ).start()
    
    def process_ai_question(self, question):
        """处理AI问题"""
        try:
            if not self.llm:
                raise RuntimeError("AI模型未初始化")
            
            response = self.llm.generate(question)
            self.root.after(0, lambda resp=response: self.append_ai_message("AI助手", resp))
        except Exception as e:
            error_msg = str(e)  # 捕获异常信息到局部变量
            self.root.after(0, lambda msg=error_msg: self.append_ai_message("系统", f"处理问题时出错: {msg}"))
        finally:
            self.root.after(0, lambda: self.ai_ask_btn.config(state=tk.NORMAL))
    
    def append_ai_message(self, sender, message):
        """添加AI对话消息"""
        self.ai_chat_text.config(state=tk.NORMAL)
        self.ai_chat_text.insert(tk.END, f"{sender}: {message}\n\n")
        self.ai_chat_text.config(state=tk.DISABLED)
        self.ai_chat_text.see(tk.END)
            
    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture("/dev/video11")
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")
                
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
        """更新摄像头画面并进行实时检测"""
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
                
                # 进行实时检测
                processed_frame = self.process_frame(frame.copy())
                
                # 转换为Tkinter可显示的格式
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
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

    def add_to_spreadsheet(self):
        """将检测结果添加到电子表单"""
        if not self.detection_results:
            messagebox.showwarning("警告", "没有可导出的检测结果")
            return
        
        try:
            with open(self.spreadsheet_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                for result in self.detection_results:
                    writer.writerow([
                        timestamp,
                        result['class'],
                        f"{result['score']:.4f}",
                        f"{result['box']}"
                    ])
            
            messagebox.showinfo("成功", f"已成功将{len(self.detection_results)}条记录添加到电子表单")
            self.status_bar.config(text=f"结果已保存到: {self.spreadsheet_path}")
            
            # 打开电子表单文件
            if messagebox.askyesno("打开文件", "是否要现在打开电子表单文件？"):
                if sys.platform == "win32":
                    os.startfile(self.spreadsheet_path)
                else:
                    os.system(f"xdg-open {self.spreadsheet_path}")
                    
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")
                
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
                self.export_btn.config(state=tk.NORMAL)
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

        # 检测完成后启用导出按钮
        if self.detection_results:
            self.export_btn.config(state=tk.NORMAL)
        else:
            self.export_btn.config(state=tk.DISABLED)
                    
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
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
    parser.add_argument('--target', type=str, default='rk3588')
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
    main()
