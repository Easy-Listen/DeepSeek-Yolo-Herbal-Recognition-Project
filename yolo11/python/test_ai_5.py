import os
import subprocess
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading

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
        print("正在初始化RKLLM运行时，请等待...")
        output = []
        while True:
            line = self.process.stdout.readline()
            if not line and self.process.poll() is not None:
                break
            if "rkllm init success" in line:
                output.append(line)
                print("模型初始化成功！")
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


class DeepseekRKLLMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepseek RKLLM 交互界面")
        self.root.geometry("800x600")
        
        self.llm = None
        self.initialization_thread = None
        self.is_initializing = False
        
        self.create_widgets()
    
    def create_widgets(self):
        # 初始化按钮
        self.init_button = tk.Button(
            self.root,
            text="初始化模型",
            command=self.start_initialization,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12)
        )
        self.init_button.pack(pady=10)
        
        # 状态标签
        self.status_label = tk.Label(
            self.root,
            text="模型未初始化",
            fg="red",
            font=("Arial", 12)
        )
        self.status_label.pack()
        
        # 输入框
        tk.Label(self.root, text="输入问题:").pack(pady=5)
        self.input_text = tk.Text(self.root, height=5, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, padx=20, pady=5)
        
        # 提问按钮
        self.ask_button = tk.Button(
            self.root,
            text="提问",
            command=self.ask_question,
            state=tk.DISABLED,
            bg="#2196F3",
            fg="white",
            font=("Arial", 12)
        )
        self.ask_button.pack(pady=5)
        
        # 输出区域
        tk.Label(self.root, text="模型回答:").pack(pady=5)
        self.output_text = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # 底部状态栏
        self.bottom_status = tk.Label(
            self.root,
            text="准备就绪",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.bottom_status.pack(fill=tk.X)
    
    def start_initialization(self):
        if self.is_initializing:
            messagebox.showinfo("提示", "模型正在初始化中，请稍候...")
            return
            
        self.is_initializing = True
        self.init_button.config(state=tk.DISABLED)
        self.status_label.config(text="模型初始化中...", fg="orange")
        self.bottom_status.config(text="正在初始化模型，这可能需要几分钟时间...")
        
        # 在后台线程中初始化模型
        self.initialization_thread = threading.Thread(
            target=self.initialize_model,
            daemon=True
        )
        self.initialization_thread.start()
        
        # 定期检查初始化状态
        self.root.after(100, self.check_initialization_status)
    
    def initialize_model(self):
        try:
            self.llm = DeepseekRKLLM()  # 调用 DeepseekRKLLM 初始化 subprocess
            self.is_initializing = False
        except Exception as e:
            self.is_initializing = False
            self.root.after(0, lambda: messagebox.showerror("初始化失败", str(e)))
    
    def check_initialization_status(self):
        if self.is_initializing:
            self.root.after(100, self.check_initialization_status)
        else:
            if self.llm is not None:
                self.status_label.config(text="模型已就绪", fg="green")
                self.ask_button.config(state=tk.NORMAL)
                self.bottom_status.config(text="模型初始化完成，可以开始提问")
                messagebox.showinfo("成功", "模型初始化完成！")
            else:
                self.status_label.config(text="模型未初始化", fg="red")
                self.init_button.config(state=tk.NORMAL)
                self.bottom_status.config(text="准备就绪")
    
    def ask_question(self):
        question = self.input_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("警告", "请输入问题")
            return
        
        self.ask_button.config(state=tk.DISABLED)
        self.input_text.config(state=tk.DISABLED)
        self.bottom_status.config(text="正在处理问题...")
        
        # 在后台线程中处理问题
        threading.Thread(
            target=self.process_question,
            args=(question,),
            daemon=True
        ).start()
    
    def process_question(self, question):
        try:
            response = self.llm.generate(question)  # 调用 subprocess 交互
            
            # 在 GUI 线程中更新界面
            self.root.after(0, lambda: self.display_response(response))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"处理问题时出错: {str(e)}"))
        finally:
            self.root.after(0, self.enable_input)
    
    def display_response(self, response):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, "问题: " + self.input_text.get("1.0", tk.END).strip() + "\n")
        self.output_text.insert(tk.END, "回答: " + response + "\n\n")
        self.output_text.config(state=tk.DISABLED)
        self.output_text.see(tk.END)
        self.bottom_status.config(text="回答完成")
    
    def enable_input(self):
        self.input_text.delete("1.0", tk.END)
        self.input_text.config(state=tk.NORMAL)
        self.ask_button.config(state=tk.NORMAL)
        self.bottom_status.config(text="准备就绪")


if __name__ == "__main__":
    root = tk.Tk()
    app = DeepseekRKLLMGUI(root)
    root.mainloop()
