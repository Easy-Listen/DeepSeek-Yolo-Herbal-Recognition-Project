import os
import cv2
import sys
import argparse
import numpy as np
import time
import random
from rknn.api import RKNN

# 中药材类别配置
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

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))  # 为每个类别生成随机颜色

def parse_args():
    parser = argparse.ArgumentParser(description='RKNN模型中药材检测测试')
    parser.add_argument('--model_path', type=str, required=True, help='RKNN模型路径')
    parser.add_argument('--img_path', type=str, required=True, help='测试图片路径或文件夹')
    parser.add_argument('--target', type=str, default='rk3588', help='目标平台')
    parser.add_argument('--img_size', type=int, default=640, help='模型输入尺寸')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--show', action='store_true', help='是否显示检测结果')
    parser.add_argument('--debug', type=int, default=1, choices=[0,1,2,3], help='调试级别 (0-3)')
    return parser.parse_args()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), debug_level=1):
    """保持长宽比的resize + padding"""
    shape = im.shape[:2]  # 当前形状 [height, width]
    
    if debug_level >= 1:
        print(f"\n[Letterbox] 原始尺寸: {shape[1]}x{shape[0]} (WxH)")
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # 左右padding
    dh /= 2  # 上下padding
    
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh)), int(round(dh))
    left, right = int(round(dw)), int(round(dw))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    if debug_level >= 1:
        print(f"处理后尺寸: {im.shape[1]}x{im.shape[0]}")
        print(f"缩放比例: {r:.4f}, 填充量: (dw={dw:.1f}, dh={dh:.1f})")
    
    return im, r, (dw, dh)

def xywh2xyxy(x):
    """将中心坐标+宽高转换为角点坐标"""
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, debug_level=1):
    """修正版NMS处理"""
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # 类别数
    output = [np.zeros((0, 6))] * bs
    
    if debug_level >= 2:
        print(f"\n[NMS] 输入形状: {prediction.shape}")
        print(f"值范围: [{prediction.min():.3f}, {prediction.max():.3f}]")
    
    for xi, x in enumerate(prediction):
        # 过滤低置信度
        x = x[x[..., 4] > conf_thres]
        
        if not x.shape[0]:
            continue
            
        # 计算类别分数 (obj_conf * cls_conf)
        x[:, 5:] *= x[:, 4:5]
        
        # 转换到图像坐标 (xywh to xyxy)
        box = xywh2xyxy(x[:, :4])
        
        # 获取最高分数类别
        conf = np.max(x[:, 5:], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:], axis=1, keepdims=True)
        x = np.concatenate([box, conf, j], axis=1)
        x = x[conf.reshape(-1) > conf_thres]
        
        # 按置信度降序排序
        x = x[x[:, 4].argsort()[::-1]]
        
        # 执行NMS
        boxes = x[:, :4].tolist()
        scores = x[:, 4].tolist()
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
        
        if len(indices) > 0:
            output[xi] = x[indices.flatten()]
            
        if debug_level >= 2:
            print(f"  NMS前/后: {len(x)} -> {len(output[xi])} 个框")
    
    return output

def scale_boxes(boxes, shape, gain, pad, debug_level=1):
    """带调试的坐标转换"""
    if debug_level >= 2:
        print(f"\n[Scale] 输入坐标样例:\n{boxes[:3]}")
        print(f"原始图像shape: {shape} | gain: {gain} | pad: {pad}")
    
    orig_boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]  # x
    boxes[:, [1, 3]] -= pad[1]  # y
    boxes[:, :4] /= gain
    
    # 裁剪到图像边界
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y
    
    if debug_level >= 2:
        print(f"转换后坐标样例:\n{boxes[:3]}")
        print(f"有效范围: x[{boxes[:, 0].min():.1f}-{boxes[:, 2].max():.1f}] "
              f"y[{boxes[:, 1].min():.1f}-{boxes[:, 3].max():.1f}]")
    
    return boxes

def draw_results(img, boxes, scores, classes, CLASSES=None, COLORS=None, debug_level=1):
    """绘制检测结果（兼容异常类别处理）"""
    if CLASSES is None:
        CLASSES = ["未知"] * (max(classes)+1) if len(classes) > 0 else ["未知"]
    if COLORS is None:
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    if debug_level >= 1:
        print(f"\n[Draw] 检测到 {len(boxes)} 个目标")
        print(f"类别分布: {np.bincount(classes)}")
    
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)
        
        # 处理异常类别
        if cls >= len(COLORS) or cls < 0:
            color = (0, 0, 255)  # 红色标记异常
            cls = min(cls, len(CLASSES)-1)
        else:
            color = COLORS[cls]
        
        # 处理类别标签
        label = f"{CLASSES[cls]}:{score:.2f}" if cls < len(CLASSES) else f"未知{cls}:{score:.2f}"
        
        # 绘制
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if debug_level >= 2:
            print(f"  {label} @ [{x1},{y1},{x2},{y2}]")
    
    return img

def load_model(model_path, target, debug_level=1):
    """加载RKNN模型"""
    rknn = RKNN()
    
    if debug_level >= 1:
        print('\n[Model] 加载模型中...')
    
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print('[ERROR] 模型加载失败!')
        exit(ret)
        
    if debug_level >= 1:
        print('[Model] 初始化运行时...')
    
    ret = rknn.init_runtime(target=target)
    if ret != 0:
        print('[ERROR] 运行时初始化失败!')
        exit(ret)
    
    if debug_level >= 1:
        print('[Model] 模型加载完成')

    
    return rknn

def detect_image(model, img_path, img_size, conf_thres, iou_thres, CLASSES=None, debug_level=1):
    """完整的检测流程"""
    # 初始化调试信息
    debug_info = {
        'timestamp': time.strftime("%Y%m%d-%H%M%S"),
        'img_path': img_path
    }
    
    if debug_level >= 1:
        print(f"\n{'='*40} 检测开始 {debug_info['timestamp']} {'='*40}")
        print(f"[Config] 尺寸: {img_size} | 置信度: {conf_thres} | IOU: {iou_thres}")

    # 图像读取
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] 无法读取图像: {img_path}") if debug_level >= 0 else None
        return None, None, None, debug_info
    
    debug_info['original_size'] = img.shape[:2]
    if debug_level >= 1:
        print(f"[1/5] 图像读取 | 尺寸: {img.shape[1]}x{img.shape[0]}")

    # 预处理
    img_letter, gain, pad = letterbox(img.copy(), new_shape=img_size, debug_level=debug_level)
    img_rgb = cv2.cvtColor(img_letter, cv2.COLOR_BGR2RGB)
    
    debug_info['preprocess'] = {
        'letterbox_size': img_letter.shape[:2],
        'gain': float(gain),
        'pad': (float(pad[0]), float(pad[1]))
    }

    # 推理
    start_time = time.time()
    try:
        outputs = model.inference(inputs=[img_rgb])
    except Exception as e:
        print(f"[ERROR] 推理失败: {str(e)}") if debug_level >= 0 else None
        return None, None, None, debug_info
    
    inference_time = time.time() - start_time
    debug_info['inference'] = {
        'time_ms': round(inference_time * 1000, 1),
        'output_shape': outputs[0].shape,
        'value_range': (float(outputs[0].min()), float(outputs[0].max()))
    }
    
    if debug_level >= 1:
        print(f"[2/5] 推理完成 | 耗时: {inference_time*1000:.1f}ms")
        print(f"输出形状: {outputs[0].shape}")

    # 后处理
    pred = non_max_suppression(outputs[0], conf_thres, iou_thres, debug_level)
    if len(pred[0]) == 0:
        print("[WARN] 未检测到目标") if debug_level >= 1 else None
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0), debug_info
    
    boxes = pred[0][:, :4]
    scores = pred[0][:, 4]
    classes = pred[0][:, 5].astype(int)
    
    debug_info['detection'] = {
        'num_objects': len(boxes),
        'classes': classes.tolist(),
        'scores': scores.tolist()
    }

    if debug_level >= 1:
        print(f"[3/5] 后处理 | 检测到 {len(boxes)} 个目标")
        if debug_level >= 2 and len(boxes) > 0:
            print("前5个检测框:")
            for i in range(min(5, len(boxes))):
                cls_name = CLASSES[classes[i]] if (CLASSES and classes[i] < len(CLASSES)) else str(classes[i])
                print(f"  {i+1}: {cls_name} | conf: {scores[i]:.3f} | box: {boxes[i].astype(int)}")

    # 坐标转换
    if len(boxes) > 0:
        boxes = scale_boxes(boxes.copy(), img.shape[:2], gain, pad, debug_level)
        debug_info['boxes'] = boxes.tolist()
        
        if debug_level >= 2:
            print(f"[4/5] 坐标转换验证:")
            print(f"有效范围: x[{boxes[:, 0].min()}-{boxes[:, 0].max()}] "
                  f"y[{boxes[:, 1].min()}-{boxes[:, 1].max()}]")

    # 可视化
    if debug_level >= 3:
        debug_img = img.copy()
        debug_img = draw_results(debug_img, boxes, scores, classes, CLASSES, debug_level=debug_level)
        cv2.imwrite(f"debug_{debug_info['timestamp']}_result.jpg", debug_img)
    
    if debug_level >= 1:
        print(f"{'='*40} 检测完成 {'='*40}")
    
    return boxes, scores, classes, debug_info

def main():
    args = parse_args()
    
    # 创建结果目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.model_path, args.target, debug_level=args.debug)

    # 获取测试图像
    if os.path.isfile(args.img_path):
        img_paths = [args.img_path]
    else:
        img_paths = [os.path.join(args.img_path, f) for f in os.listdir(args.img_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 处理每张图像
    for img_path in img_paths:
        print(f"\nProcessing: {img_path}")
        
        # 检测
        boxes, scores, classes, debug_info = detect_image(
            model, img_path, args.img_size, 
            args.conf_thres, args.iou_thres,
            CLASSES=CLASSES,
            debug_level=args.debug
        )

        # 读取原始图像用于绘制
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 绘制结果
        if boxes is not None and len(boxes) > 0:
            img = draw_results(img, boxes, scores, classes, CLASSES, COLORS, debug_level=args.debug)
        
        # 保存结果
        save_path = os.path.join(args.save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img)
        print(f"Result saved to: {save_path}")
        
        # 显示结果
        if args.show:
            cv2.imshow('Detection Result', img)
            cv2.waitKey(0)
    
    # 释放模型
    model.release()
    print("\nDone!")

if __name__ == '__main__':
    main()
