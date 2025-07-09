import threading
import time
from queue import Queue

# 线程状态反馈队列
status_queue = Queue()

def worker(worker_id, task_time):
    """工作线程函数"""
    status_queue.put(f"工人{worker_id}: 开始工作")
    time.sleep(task_time)  # 模拟工作时间
    status_queue.put(f"工人{worker_id}: 工作完成 (耗时{task_time}秒)")

def monitor():
    """监控线程函数，用于实时反馈状态"""
    while True:
        status = status_queue.get()
        if status == "STOP":
            print("监控线程结束")
            break
        print(f"[状态更新] {status}")
        status_queue.task_done()

def main():
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    # 创建3个工作线程
    workers = []
    for i in range(1, 4):
        t = threading.Thread(target=worker, args=(i, i*2))
        workers.append(t)
        t.start()

    # 等待所有工作线程完成
    for t in workers:
        t.join()

    # 通知监控线程结束
    status_queue.put("STOP")
    monitor_thread.join()

    print("所有任务完成！")

if __name__ == "__main__":
    main()
