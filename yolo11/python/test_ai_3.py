import os
import subprocess
import time

class DeepseekRKLLM:
    def __init__(self, model_path, exec_path, lib_path, init_timeout=120):
        """
        :param model_path: RKLLM模型文件路径
        :param exec_path: 可执行文件路径
        :param lib_path: 动态库目录
        :param init_timeout: 初始化超时时间（秒）
        """
        self.model_path = os.path.abspath(model_path)
        self.exec_path = os.path.abspath(exec_path)
        self.lib_path = os.path.abspath(lib_path)
        self.init_timeout = init_timeout
        self.process = None
        
        # 添加路径验证
        self._validate_paths()
        self._init_process()

    def _validate_paths(self):
        """校验文件是否存在"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if not os.path.exists(self.exec_path):
            raise FileNotFoundError(f"可执行文件不存在: {self.exec_path}")
        if not os.access(self.exec_path, os.X_OK):
            raise PermissionError(f"可执行文件没有权限: {self.exec_path}")

    def _init_process(self):
        """启动子进程并等待初始化完成"""
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{self.lib_path}:" + env.get("LD_LIBRARY_PATH", "")

        # 启动子进程
        self.process = subprocess.Popen(
            [self.exec_path, self.model_path, "128", "512"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # 等待初始化完成
        print(f"等待模型加载，最多等待 {self.init_timeout} 秒...")
        start_time = time.time()
        while time.time() - start_time < self.init_timeout:
            if self._check_init_done():
                print("模型加载完成！")
                return        
        raise RuntimeError("模型初始化超时")

    def _check_init_done(self):
        """检查子进程是否初始化完成"""
        # 这里假设C++程序初始化完成后会输出"init done"
        line = self.process.stdout.readline()
        return "用户" in line if line else False

    def generate(self, prompt, max_new_tokens=128, max_context_len=128):
        """发送请求并获取响应"""
        if not self.process:
            raise RuntimeError("子进程未启动")

        # 构造输入
        formatted_prompt = (
            "<|im_start|>system\n你是一名专业AI助手<|im_end|>\n"
            "<|im_start|>user\n" + prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # 发送请求
        self.process.stdin.write(formatted_prompt + "\n")
        self.process.stdin.flush()

        # 获取响应
        output = []
        start_time = time.time()
        while time.time() - start_time < 60:  # 响应超时60秒
            line = self.process.stdout.readline()
            if not line or "<|im_end|>" in line:
                break
            output.append(line)

        return "".join(output).strip()

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'process') and self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
            except:
                pass

# 使用示例
if __name__ == "__main__":
    try:
        llm = DeepseekRKLLM(
            model_path="/media/elf/EXT4/rkllm/deepseek-r1-1.5b-w8a8.rkllm",
            exec_path="/media/elf/EXT4/rkllm/llm_demo_3",
            lib_path="/media/elf/EXT4/rkllm/lib",
            init_timeout=120
        )

        while True:
            question = input("\n用户: ")
            if question.lower() in ("exit", "quit"):
                break
            
            try:
                answer = llm.generate(question)
                print(f"\n助手: {answer}")
            except Exception as e:
                print(f"错误: {e}")
                
    except Exception as e:
        print(f"初始化失败: {e}")
