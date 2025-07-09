import os
import cv2
import sys
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

from py_utils.coco_utils import COCO_test_helper
import numpy as np


OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = (
    "bai fu ling",    # 白茯苓
    "bai shao",       # 白芍
    "bai zhu",        # 白术
    "pu gong ying",   # 蒲公英
    "gan cao",        # 甘草
    "zhi zi",         # 栀子
    "dang shen",      # 党参
    "tao ren",        # 桃仁
    "qu pi tao ren",  # 去皮桃仁
    "di fu zi",       # 地肤子
    "mu dan pi",      # 牡丹皮
    "dong chong xia cao",  # 冬虫夏草
    "du zhong",       # 杜仲
    "dang gui",       # 当归
    "xing ren",       # 杏仁
    "he shou wu",     # 何首乌
    "huang jing",     # 黄精
    "ji xue teng",    # 鸡血藤
    "gou qi",         # 枸杞
    "lian xu",        # 莲须
    "lian rou",       # 莲肉
    "mai men dong",   # 麦门冬
    "mu tong",        # 木通
    "yu zhu",         # 玉竹
    "nv zhen zi",     # 女贞子
    "rou cong rong",  # 肉苁蓉
    "ren shen",       # 人参
    "wu mei",         # 乌梅
    "fu pen zi",      # 覆盆子
    "gua lou pi",     # 瓜蒌皮
    "rou gui",        # 肉桂
    "shan zhu yu",    # 山茱萸
    "shan yao",       # 山药
    "suan zao ren",   # 酸枣仁
    "sang bai pi",    # 桑白皮
    "shan zha",       # 山楂
    "tian ma",        # 天麻
    "shu di huang",   # 熟地黄
    "xiao hui xiang", # 小茴香
    "ze xie",         # 泽泻
    "zhu ru",         # 竹茹
    "chuan bei mu",   # 川贝母
    "chuan xiong",    # 川芎
    "xuan shen",      # 玄参
    "yi zhi ren"      # 益智仁
)


coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
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
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
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

def dfl(position):
    # Distribution Focal Loss (DFL)
    import torch
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
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

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
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
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

class VideoApp:
    def __init__(self, root, model, platform, args):
        self.root = root
        self.model = model
        self.platform = platform
        self.args = args
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        self.cap = cv2.VideoCapture(0)  # 默认摄像头
        self.is_running = False
        self.capture_flag = False
        self.image_files = ['1.jpg', '2.jpg', '3.jpg']
        # 设置窗口
        self.root.title("中药材识别系统")
        
        # 视频显示区域
        self.video_label = ttk.Label(root)
        self.video_label.pack(padx=10, pady=10)
        
        # 控制按钮区域
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(self.control_frame, text="开始", command=self.start_video)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(self.control_frame, text="停止", command=self.stop_video)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_btn = ttk.Button(self.control_frame, text="拍照", command=self.capture_image)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        self.exit_btn = ttk.Button(self.control_frame, text="退出", command=self.exit_app)
        self.exit_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态标签
        self.status_label = ttk.Label(root, text="准备就绪")
        self.status_label.pack(pady=5)
        
        # 结果显示区域
        self.result_text = tk.Text(root, height=10, width=50)
        self.result_text.pack(pady=10)
        self.result_text.insert(tk.END, "识别结果将显示在这里...\n")
        self.result_text.config(state=tk.DISABLED)
        
    def start_video(self):
        if not self.is_running:
            self.is_running = True
            self.update_video()
            self.status_label.config(text="摄像头已开启")
    
    def stop_video(self):
        self.is_running = False
        self.status_label.config(text="摄像头已停止")
    
    def capture_image(self):
        self.capture_flag = True
        self.status_label.config(text="拍照请求已发送")
    
    def exit_app(self):
        self.is_running = False
        self.cap.release()
        self.root.destroy()
    
    def process_frame(self, frame):
        # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
        pad_color = (0,0,0)
        img = self.co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocess if not rknn model
        if self.platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2,0,1))
            input_data = input_data.reshape(1,*input_data.shape).astype(np.float32)
            input_data = input_data/255.
        else:
            input_data = img

        outputs = self.model.run([input_data])
        boxes, classes, scores = post_process(outputs)
        
        if boxes is not None:
            draw(frame, self.co_helper.get_real_box(boxes), scores, classes)
        
        return frame
    
    def save_image(self, frame):
        if not os.path.exists('./captures'):
            os.mkdir('./captures')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./captures/capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        self.status_label.config(text=f"照片已保存: {filename}")
    
    def update_video(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # 处理帧
                processed_frame = self.process_frame(frame)
                
                # 如果需要拍照
                if self.capture_flag:
                    self.save_image(processed_frame)
                    self.capture_flag = False
                
                # 转换为Tkinter可显示的格式
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(image=img)
                
                # 更新显示
                self.video_label.img = img
                self.video_label.config(image=img)
            
            # 继续更新
            self.root.after(10, self.update_video)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    
    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

    # data params
    # parser.add_argument('--anno_json', type=str, default='../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
    # coco val folder: '../../../datasets/COCO//val2017'
    parser.add_argument('--img_folder', type=str, default='../model', help='img folder path')
    parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')

    args = parser.parse_args()

    # init model
    model, platform = setup_model(args)
    
    # 检查是否要运行GUI界面
    if args.img_show:
        # 创建GUI界面
        root = tk.Tk()
        app = VideoApp(root, model, platform, args)
        root.mainloop()
    else:
        # 运行原始图像处理流程
        file_list = sorted(os.listdir(args.img_folder))
        img_list = []
        for path in file_list:
            if img_check(path):
                img_list.append(path)
        co_helper = COCO_test_helper(enable_letter_box=True)

        # run test
        for i in range(len(img_list)):
            print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

            img_name = img_list[i]
            img_path = os.path.join(args.img_folder, img_name)
            if not os.path.exists(img_path):
                print("{} is not found", img_name)
                continue

            img_src = cv2.imread(img_path)
            if img_src is None:
                continue

            # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
            pad_color = (0,0,0)
            img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # preprocee if not rknn model
            if platform in ['pytorch', 'onnx']:
                input_data = img.transpose((2,0,1))
                input_data = input_data.reshape(1,*input_data.shape).astype(np.float32)
                input_data = input_data/255.
            else:
                input_data = img

            outputs = model.run([input_data])
            boxes, classes, scores = post_process(outputs)

            if args.img_show or args.img_save:
                print('\n\nIMG: {}'.format(img_name))
                img_p = img_src.copy()
                if boxes is not None:
                    draw(img_p, co_helper.get_real_box(boxes), scores, classes)

                if args.img_save:
                    if not os.path.exists('./result'):
                        os.mkdir('./result')
                    result_path = os.path.join('./result', img_name)
                    cv2.imwrite(result_path, img_p)
                    print('Detection result save to {}'.format(result_path))
                            
                if args.img_show:
                    cv2.imshow("full post process result", img_p)
                    cv2.waitKeyEx(0)

            # record maps
            if args.coco_map_test is True:
                if boxes is not None:
                    for i in range(boxes.shape[0]):
                        co_helper.add_single_record(image_id = int(img_name.split('.')[0]),
                                                    category_id = coco_id_list[int(classes[i])],
                                                    bbox = boxes[i],
                                                    score = round(scores[i], 5).item()
                                                    )

        # calculate maps
        if args.coco_map_test is True:
            pred_json = args.model_path.split('.')[-2]+ '_{}'.format(platform) +'.json'
            pred_json = pred_json.split('/')[-1]
            pred_json = os.path.join('./', pred_json)
            co_helper.export_to_json(pred_json)

            from py_utils.coco_utils import coco_eval_with_json
            coco_eval_with_json(args.anno_json, pred_json)

    # release
    model.release()

if __name__ == '__main__':
    main()

