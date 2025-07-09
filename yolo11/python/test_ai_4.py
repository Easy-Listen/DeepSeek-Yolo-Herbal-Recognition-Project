import os
import subprocess
import time
RESPONSE_TERMINATOR = "<|endoftext|>"  # 必须与C++完全一致

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
        
        # 准备命令和环境变量
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
        
        # 等待初始化（根据硬件性能调整）
        print("正在初始化RKLLM运行时，请等待...")
        # time.sleep(120)  # 实际使用时可根据需要调整
        
        # 获取输出
        output = []
        while True:
            line = self.process.stdout.readline()
            if not line and self.process.poll() is not None:
                break
            if "rkllm init success" in line:
                output.append(line)
                print("模型初始化成功！")  # 实时打印输出
                break

        return "".join(output) if output else "无响应"

    def generate(self, prompt):
        """生成文本（单次交互）"""
        if not self.process:
            raise RuntimeError("子进程未启动")

        # 发送输入（确保以换行符结尾）
        self.process.stdin.write(f"{prompt}\n")
        self.process.stdin.flush()

        # 收集输出
        output = []
        while True:
            line = self.process.stdout.readline()
            if not line:  # 管道关闭
                break
            line = line.strip()
            if line:  # 忽略空行
                output.append(line)
                print(f"[Response] {line}")  # 实时输出
            # 关键修改：检测终止符
            if line.endswith(RESPONSE_TERMINATOR):
                output[-1] = line[:-len(RESPONSE_TERMINATOR)].strip()  # 移除终止符
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

if __name__ == "__main__":
    try:
        llm = DeepseekRKLLM()
        
        while True:
            question = input("\n请输入问题 (输入q退出): ").strip()
            if question.lower() == 'q':
                break
            
            try:
                start_time = time.time()
                response = llm.generate(question)
            except:
                pass
                
    except Exception as e:
        print(f"初始化失败: {str(e)}")
