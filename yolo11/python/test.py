import cv2
import os
import datetime

# 创建保存图片的目录
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# 使用video11 (ISP-processed feed)
cap = cv2.VideoCapture("/dev/video11")

# 设置分辨率 (例如1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 设置格式 (NV12是常见格式)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('N', 'V', '1', '2'))

print("按 's' 键拍照，按 'q' 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧，退出...")
        break
    
    # 如果需要，将YUV (NV12)转换为BGR
    # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
    
    cv2.imshow("Camera Feed", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按q退出
        break
    elif key == ord('s'):  # 按s拍照
        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")
        
        # 保存图片
        if cv2.imwrite(filename, frame):
            print(f"图片已保存为: {filename}")
        else:
            print("保存图片失败!")

cap.release()
cv2.destroyAllWindows()
