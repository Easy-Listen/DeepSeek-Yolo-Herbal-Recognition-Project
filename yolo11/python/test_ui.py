import sys
sys.path.append("/media/elf/EXT4/python_libs")
from ttkbootstrap import Style
from ttkbootstrap.constants import *
import ttkbootstrap as ttk
import tkinter as tk

root = ttk.Window(themename="cyborg")  # 黑灰+绿，科技感
style = Style()

frame = ttk.Frame(root, padding=20)
frame.pack()

ttk.Label(frame, text="AI药材识别系统", font=("Helvetica", 16)).pack(pady=10)
ttk.Button(frame, text="开始识别", bootstyle="success").pack(pady=5)
ttk.Button(frame, text="导出结果", bootstyle="info-outline").pack(pady=5)

root.mainloop()
