import os
import cv2
import sys
import argparse
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

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

'''
def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
'''
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
    image[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
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

def detect_image_list(model, platform, args):
    file_list = sorted(os.listdir(args.img_folder))
    img_list = [f for f in file_list if img_check(f)]
    co_helper = COCO_test_helper(enable_letter_box=True)

    for i, img_name in enumerate(img_list):
        print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print(f"{img_name} not found.")
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2, 0, 1)).reshape(1, 3, IMG_SIZE[1], IMG_SIZE[0]).astype(np.float32) / 255.0
        else:
            input_data = img

        outputs = model.run([input_data])
        boxes, classes, scores = post_process(outputs)

        if args.img_show or args.img_save:
            print('\n\nIMG:', img_name)
            img_p = img_src.copy()
            if boxes is not None:
                draw(img_p, co_helper.get_real_box(boxes), scores, classes)

            if args.img_save:
                os.makedirs('./result', exist_ok=True)
                result_path = os.path.join('./result', img_name)
                cv2.imwrite(result_path, img_p)
                print('Detection result saved to', result_path)

            if args.img_show:
                cv2.imshow("Detection Result", img_p)
                cv2.waitKeyEx(0)

        if args.coco_map_test and boxes is not None:
            for j in range(boxes.shape[0]):
                co_helper.add_single_record(
                    image_id=int(img_name.split('.')[0]),
                    category_id=coco_id_list[int(classes[j])],
                    bbox=boxes[j],
                    score=round(scores[j], 5).item()
                )

    if args.coco_map_test:
        pred_json = f"{args.model_path.split('.')[-2]}_{platform}.json"
        pred_json = os.path.join('./', os.path.basename(pred_json))
        co_helper.export_to_json(pred_json)

        from py_utils.coco_utils import coco_eval_with_json
        coco_eval_with_json(args.anno_json, pred_json)
def detect_camera(model, platform, args):
    """Real-time camera detection with photo capture functionality"""
    # Initialize camera
    cap = cv2.VideoCapture("/dev/video11")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('N', 'V', '1', '2'))
    
    # Create directory for saved images
    save_dir = "captured_images"
    os.makedirs(save_dir, exist_ok=True)
    
    co_helper = COCO_test_helper(enable_letter_box=True)
    
    print("Press 's' to capture and save image, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Process frame for detection
        img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2, 0, 1)).reshape(1, 3, IMG_SIZE[1], IMG_SIZE[0]).astype(np.float32) / 255.0
        else:
            input_data = img
        
        # Run detection
        outputs = model.run([input_data])
        boxes, classes, scores = post_process(outputs)
        
        # Draw detection results on display frame
        display_frame = frame.copy()
        if boxes is not None:
            real_boxes = co_helper.get_real_box(boxes)
            draw(display_frame, real_boxes, scores, classes)
        
        # Show frame
        cv2.imshow("Camera Detection", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Save image
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")
            
            # Save original frame with detection results
            if cv2.imwrite(filename, display_frame):
                print(f"Image saved: {filename}")
                
                # Also save the detection results to a text file
                result_txt = os.path.join(save_dir, f"capture_{timestamp}.txt")
                with open(result_txt, 'w') as f:
                    if boxes is not None:
                        for box, score, cl in zip(real_boxes, scores, classes):
                            f.write(f"{CLASSES[cl]}: {score:.3f} @ {box}\n")
                    else:
                        f.write("No objects detected\n")
                print(f"Detection results saved: {result_txt}")
            else:
                print("Failed to save image")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--target', type=str, default='rk3566')
    parser.add_argument('--device_id', type=str, default=None)
    parser.add_argument('--img_show', action='store_true')
    parser.add_argument('--img_save', action='store_true')
    parser.add_argument('--img_folder', type=str, default='../model')
    parser.add_argument('--coco_map_test', action='store_true')
    parser.add_argument('--anno_json', type=str, default=None)
    parser.add_argument('--camera', action='store_true', help='Enable camera detection mode')

    args = parser.parse_args()

    model, platform = setup_model(args)
    
    if args.camera:
        detect_camera(model, platform, args)
    else:
        detect_image_list(model, platform, args)
    
    model.release()
