import os
import subprocess
import time

class DeepseekRKLLM:
    def __init__(self):
        # 模型和可执行文件路径（需根据实际路径修改）
        self.deepseek_rkllm_path = "/media/elf/EXT4/rkllm/deepseek-r1-1.5b-w8a8.rkllm"
        self.llm_demo_path = "/media/elf/EXT4/rkllm/llm_demo_3"
        self.lib_path = "/media/elf/EXT4/rkllm/lib"

    def _validate_paths(self):
        """检查必要文件是否存在"""
        if not os.path.exists(self.deepseek_rkllm_path):
            raise FileNotFoundError(f"模型文件不存在: {self.deepseek_rkllm_path}")
        if not os.path.exists(self.llm_demo_path):
            raise FileNotFoundError(f"可执行文件不存在: {self.llm_demo_path}")
        if not os.access(self.llm_demo_path, os.X_OK):
            raise PermissionError(f"可执行文件没有权限: {self.llm_demo_path}")

    def generate(self, prompt, max_new_tokens=128, num_beams=64):
        """
        调用RKLLM生成文本
        :param prompt: 输入提示
        :param max_new_tokens: 最大生成长度
        :param num_beams: Beam Search参数
        :return: 生成的文本
        """
        self._validate_paths()

        # 准备命令和环境变量
        cmd = [
            self.llm_demo_path,
            self.deepseek_rkllm_path,
            str(max_new_tokens),
            str(num_beams)
        ]
        
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{self.lib_path}:" + env.get("LD_LIBRARY_PATH", "")

        # 启动子进程
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )

        # 发送输入
        # process.stdin.write(f"{prompt}\n")
        process.stdin.flush()
        process.stdin.close()

        # 等待初始化（根据硬件性能调整）
        print("正在初始化RKLLM运行时，请等待...")
        time.sleep(120)  # 实际使用时可根据需要调整

        # 获取输出
        output = []
        while True:
            line = process.stdout.readline()
            
            if not line and process.poll() is not None:
                break
            if line:
                output.append(line)
                # print("收到部分响应:", line.strip())  # 实时打印输出

        return "".join(output) if output else "无响应"

# 使用示例
if __name__ == "__main__":
    llm = DeepseekRKLLM()
    
    # 测试连接
    try:
        print("开始测试AI连接...")
        response = llm.generate("测试连接，请回复'OK'")
        print("测试结果:", response)
        question = input("\n")
        start_time = time.time()
        answer = llm.generate(question)
    except Exception as e:
        print(f"发生错误: {str(e)}")
