import os
import sys
import time
import tkinter as tk
import subprocess
from tkinter import ttk, messagebox
from threading import Thread

class DeepseekApp:
    def __init__(self, root):
        self.root = root
        self.deepseek_rkllm_path = "/media/elf/EXT4/rkllm/deepseek-r1-1.5b-w8a8.rkllm"
        self.llm_demo_path = "/media/elf/EXT4/rkllm/llm_demo_3"
        
        # 设置主窗口
        self.root.title("Deepseek-R1 本地咨询")
        self.root.geometry("800x600")   
        
        # 创建界面组件
        self.create_widgets()
        
    def create_widgets(self):
        """创建所有GUI组件"""
        # 主框架布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 问题输入框
        self.question_entry = ttk.Entry(main_frame)
        self.question_entry.pack(fill=tk.X, padx=5, pady=5)
        self.question_entry.insert(0, "请输入您的问题...")
        
        # 回答显示区域
        self.answer_text = tk.Text(main_frame, height=20)
        self.answer_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.answer_text.config(state=tk.DISABLED)
        
        # 控制按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # 提问按钮
        self.ask_button = ttk.Button(
            button_frame, 
            text="提问", 
            command=self.ask_deepseek
        )
        self.ask_button.pack(side=tk.LEFT, padx=5)
        
        # 测试按钮
        self.test_button = ttk.Button(
            button_frame,
            text="测试连接",
            command=self.test_ai_connection
        )
        self.test_button.pack(side=tk.LEFT, padx=5)
        
        # 状态栏
        self.status_bar = ttk.Label(main_frame, text="准备就绪", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(5,0))

    def update_status(self, message):
        """更新状态信息"""
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.insert(tk.END, f"{message}\n")
        self.answer_text.see(tk.END)
        self.answer_text.config(state=tk.DISABLED)
        self.root.update()  # 强制更新界面

    def execute_deepseek(self, prompt_data):
        """执行Deepseek进程"""
        try:
            # 检查文件是否存在
            if not os.path.exists(prompt_data['deepseek_rkllm_path']):
                return f"错误: 模型文件不存在 {prompt_data['deepseek_rkllm_path']}"
                
            if not os.path.exists(prompt_data['llm_demo_path']):
                return f"错误: 可执行文件不存在 {prompt_data['llm_demo_path']}"
            
            # 检查执行权限
            if not os.access(prompt_data['llm_demo_path'], os.X_OK):
                return f"错误: 可执行文件没有执行权限 {prompt_data['llm_demo_path']}"
            

            
            # 准备命令和环境变量
            cmd = [
                prompt_data['llm_demo_path'],
                prompt_data['deepseek_rkllm_path'],
                str(prompt_data['max_new_tokens']),
                str(prompt_data['num_beams'])
            ]
            
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = "/media/elf/EXT4/rkllm/lib:" + env.get("LD_LIBRARY_PATH", "")
            
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

            # 发送输入并获取输出
            process.stdin.write(f"用户：{prompt_data['prompt']}\n")
            process.stdin.flush()
            process.stdin.close()
           
            # 添加初始化等待时间
            self.root.after(0, lambda: self.update_status("正在初始化RKLLM运行时，请等待2分钟..."))
            time.sleep(120)  # 等待2分钟
            
            output = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    output.append(line)
            
            return "".join(output) if output else "无响应"
            
        except Exception as e:
            return f"执行错误: {str(e)}"

    def run_deepseek_local(self, prompt, max_new_tokens=128, num_beams=1):
        """本地模型调用"""
        try:
            # 验证路径
            if not all([
                os.path.exists(self.deepseek_rkllm_path),
                os.path.exists(self.llm_demo_path)
            ]):
                return f"错误: 模型路径或可执行文件路径无效\n模型: {self.deepseek_rkllm_path}\n可执行文件: {self.llm_demo_path}"
            
            # 准备数据
            prompt_data = {
                'deepseek_rkllm_path': self.deepseek_rkllm_path,
                'llm_demo_path': self.llm_demo_path,
                'prompt': prompt,
                'max_new_tokens': max_new_tokens,
                'num_beams': num_beams
            }
            
            return self.execute_deepseek(prompt_data)
                    
        except Exception as e:
            return f"执行错误: {str(e)}"

    def test_ai_connection(self):
        """测试AI连接是否正常"""
        if hasattr(self, '_testing_thread') and self._testing_thread.is_alive():
            messagebox.showwarning("警告", "已有测试正在进行中")
            return
            
        # 显示测试状态
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, "正在测试AI连接...\n")
        self.answer_text.config(state=tk.DISABLED)
        self.test_button.config(state=tk.DISABLED)
        
        # 启动线程
        self._testing_thread = Thread(
            target=self._threaded_test_connection,
            daemon=True
        )
        self._testing_thread.start()
        
        # 启动进度监控
        self._check_test_status()

    def _threaded_test_connection(self):
        """线程内执行的实际测试逻辑"""
        try:
            
            test_prompt = "测试连接，请回复'连接成功'"
            answer = self.run_deepseek_local(test_prompt)
            
            # 判断响应状态
            if "用户：" in answer:
                result = "AI连接测试成功"
                details = "系统已成功连接到AI模型并获取响应"
            else:
                result = "AI连接测试异常"
                details = f"收到非预期响应:\n{answer}"
                
            self.root.after(0, lambda: self._update_test_result(result, details))
        except Exception as e:
            self.root.after(0, lambda: self._update_test_result("AI连接测试异常", f"测试过程中发生错误: {str(e)}"))

    def _update_test_result(self, result, details):
        """更新测试结果到界面"""
        try:
            self.answer_text.config(state=tk.NORMAL)
            self.answer_text.delete(1.0, tk.END)
            self.answer_text.insert(tk.END, f"{result}\n{details}\n")
            self.answer_text.see(tk.END)
            
            # 根据结果显示不同颜色
            if "成功" in result:
                self.answer_text.tag_add("success", "1.0", "1.0 lineend")
                self.answer_text.tag_config("success", foreground="green")
            else:
                self.answer_text.tag_add("error", "1.0", "1.0 lineend")
                self.answer_text.tag_config("error", foreground="red")
                
        except Exception as e:
            print(f"更新测试结果异常: {str(e)}")
        finally:
            self._safe_disable_text()
            self.test_button.config(state=tk.NORMAL)

    def _check_test_status(self):
        """定期检查测试线程状态"""
        if self._testing_thread.is_alive():
            self.root.after(1000, self._check_test_status)
        else:
            self.test_button.config(state=tk.NORMAL)

    def ask_deepseek(self):
        """线程安全的提问方法"""
        if hasattr(self, '_thinking_thread') and self._thinking_thread.is_alive():
            messagebox.showwarning("警告", "已有问题正在处理中")
            return
            
        question = self.question_entry.get().strip()
        if not question or question == "请输入您的问题...":
            return
            
        # 显示状态
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, f"用户：{question}\nAI思考中...\n")
        self.answer_text.config(state=tk.DISABLED)
        self.ask_button.config(state=tk.DISABLED)
        
        # 启动线程
        self._thinking_thread = Thread(
            target=self._threaded_ask,
            args=(question,),
            daemon=True
        )
        self._thinking_thread.start()
        
        # 启动进度监控
        self._check_thread_status()

    def _threaded_ask(self, question):
        """线程内执行的实际逻辑"""
        try:
            # 获取回答
            answer = self.run_deepseek_local(question)
            self.root.after(0, lambda: self._update_answer(answer))
        except Exception as e:
            self.root.after(0, lambda: self._update_answer(f"系统错误: {str(e)}"))

    def _update_answer(self, answer):
        """更新答案到界面"""
        try:
            self.answer_text.config(state=tk.NORMAL)
            self.answer_text.insert(tk.END, f"\nAI回复:\n{answer}\n")
            self.answer_text.see(tk.END)
        except Exception as e:
            print(f"更新答案异常: {str(e)}")
        finally:
            self._safe_disable_text()

    def _check_thread_status(self):
        """定期检查线程状态"""
        if self._thinking_thread.is_alive():
            self.root.after(1000, self._check_thread_status)
        else:
            self.ask_button.config(state=tk.NORMAL)

    def _safe_disable_text(self):
        """安全禁用文本编辑"""
        if hasattr(self, 'answer_text') and self.answer_text.winfo_exists():
            self.answer_text.config(state=tk.DISABLED)

    def on_closing(self):
        """关闭窗口时的清理工作"""
        self.root.destroy()

def main():
    # 创建GUI应用
    root = tk.Tk()
    app = DeepseekApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
