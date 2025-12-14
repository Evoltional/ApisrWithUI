import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import warnings
import collections
import json
from datetime import datetime
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np
import torch
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings('ignore')

# 添加APISR项目路径
sys.path.append('.')

# 直接从architecture导入模型
try:
    from architecture.rrdb import RRDBNet
    from architecture.grl import GRL
    from architecture.dat import DAT
    from architecture.cunet import UNet_Full
except ImportError as e:
    print(f"导入模型架构时出错: {e}")
    print("请确保architecture模块在Python路径中")
    sys.exit(1)


class ModernButton(ttk.Button):
    """现代化按钮样式"""

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(style='Accent.TButton')


class APISRVideoProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("APISR 视频超分辨率处理工具")
        self.root.geometry("1200x800")

        # 设置窗口图标（如果有的话）
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass

        # 设置主题颜色
        self.bg_color = "#f5f5f7"
        self.sidebar_color = "#2c3e50"
        self.accent_color = "#3498db"
        self.success_color = "#27ae60"
        self.warning_color = "#f39c12"
        self.danger_color = "#e74c3c"

        # 配置文件路径
        self.config_file = "apisr_config.json"

        # 初始化变量
        self.input_paths = []  # 改为存储多个视频路径的列表
        self.output_dir = tk.StringVar()
        self.model_var = tk.StringVar(value="GRL")
        self.scale_var = tk.StringVar(value="4")
        self.segment_duration = tk.StringVar(value="20")
        self.downsample_threshold = tk.StringVar(value="720")
        self.float16_var = tk.BooleanVar(value=False)
        self.crop_for_4x_var = tk.BooleanVar(value=True)
        self.batch_size_var = tk.StringVar(value="1")
        self.tile_size_var = tk.StringVar(value="128")
        self.hash_threshold_var = tk.StringVar(value="3")
        self.ssim_threshold_var = tk.StringVar(value="0.98")
        self.enable_dup_detect_var = tk.BooleanVar(value=True)
        self.use_ssim_var = tk.BooleanVar(value=True)
        self.use_hash_var = tk.BooleanVar(value=True)
        self.test_mode_var = tk.BooleanVar(value=False)
        self.enable_history_var = tk.BooleanVar(value=True)
        self.history_size_var = tk.StringVar(value="20")

        # 设置样式
        self.setup_styles()

        # 模型信息
        self.models = {
            "GRL": {"scale": [4], "weight": "pretrained/4x_APISR_GRL_GAN_generator.pth"},
            "DAT": {"scale": [4], "weight": "pretrained/4x_APISR_DAT_GAN_generator.pth"},
            "RRDB": {"scale": [2, 4], "weight": {
                "2": "pretrained/2x_APISR_RRDB_GAN_generator.pth",
                "4": "pretrained/4x_APISR_RRDB_GAN_generator.pth"
            }},
            "CUNET": {"scale": [4], "weight": "pretrained/4x_APISR_CUNET_GAN_generator.pth"}
        }

        # 处理器状态
        self.processing = False
        self.paused = False
        self.stopped = False
        self.generator = None
        self.weight_dtype = torch.float32

        # 进度恢复相关
        self.current_video_index = 0  # 新增：当前处理视频索引
        self.current_segment_index = 0
        self.current_frame_in_segment = 0
        self.total_segments = 0
        self.segments = []
        self.processed_segments = []
        self.progress_data_file = None

        # 重复帧检测相关
        self.dup_frame_count = 0

        # 新增：历史帧缓存系统
        self.init_history_cache()

        # 临时文件路径
        self.temp_base_dir = None
        self.current_segment_frames_dir = None
        self.video_base_name = None

        # 线程控制
        self.processing_thread = None
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.processing_lock = threading.Lock()
        self.last_save_time = 0
        self.save_interval = 10

        # 新增：暂停时的内存优化
        self.pause_lock = threading.Lock()
        self.pause_cv = threading.Condition(self.pause_lock)
        self.should_sleep = False

        # 设置历史帧数量验证
        self.setup_history_size_validation()

        self.setup_ui()

        # 设置初始模型
        self.on_model_change()

        # 加载配置文件
        self.load_config()

        # 绑定配置保存事件
        self.setup_config_save_bindings()

    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 设置变量
                if 'model' in config:
                    self.model_var.set(config['model'])
                if 'scale' in config:
                    self.scale_var.set(str(config['scale']))
                if 'segment_duration' in config:
                    self.segment_duration.set(str(config['segment_duration']))
                if 'downsample_threshold' in config:
                    self.downsample_threshold.set(str(config['downsample_threshold']))
                if 'float16' in config:
                    self.float16_var.set(config['float16'])
                if 'crop_for_4x' in config:
                    self.crop_for_4x_var.set(config['crop_for_4x'])
                if 'batch_size' in config:
                    self.batch_size_var.set(str(config['batch_size']))
                if 'tile_size' in config:
                    self.tile_size_var.set(str(config['tile_size']))
                if 'hash_threshold' in config:
                    self.hash_threshold_var.set(str(config['hash_threshold']))
                if 'ssim_threshold' in config:
                    self.ssim_threshold_var.set(str(config['ssim_threshold']))
                if 'enable_dup_detect' in config:
                    self.enable_dup_detect_var.set(config['enable_dup_detect'])
                if 'use_ssim' in config:
                    self.use_ssim_var.set(config['use_ssim'])
                if 'use_hash' in config:
                    self.use_hash_var.set(config['use_hash'])
                if 'test_mode' in config:
                    self.test_mode_var.set(config['test_mode'])
                if 'enable_history' in config:
                    self.enable_history_var.set(config['enable_history'])
                if 'history_size' in config:
                    self.history_size_var.set(str(config['history_size']))

                self.log(f"已从 {self.config_file} 加载配置")

                # 更新UI状态
                self.on_model_change()
                self.toggle_history_settings()
            except Exception as e:
                self.log(f"加载配置文件时出错: {e}")
        else:
            self.log("未找到配置文件，使用默认配置")

    def save_config(self):
        """保存配置文件"""
        try:
            config = {
                'model': self.model_var.get(),
                'scale': int(self.scale_var.get()),
                'segment_duration': int(self.segment_duration.get()),
                'downsample_threshold': int(self.downsample_threshold.get()),
                'float16': self.float16_var.get(),
                'crop_for_4x': self.crop_for_4x_var.get(),
                'batch_size': int(self.batch_size_var.get()),
                'tile_size': int(self.tile_size_var.get()),
                'hash_threshold': int(self.hash_threshold_var.get()),
                'ssim_threshold': float(self.ssim_threshold_var.get()),
                'enable_dup_detect': self.enable_dup_detect_var.get(),
                'use_ssim': self.use_ssim_var.get(),
                'use_hash': self.use_hash_var.get(),
                'test_mode': self.test_mode_var.get(),
                'enable_history': self.enable_history_var.get(),
                'history_size': int(self.history_size_var.get()),
                'last_saved': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)

            self.log(f"配置已保存到 {self.config_file}")
        except Exception as e:
            self.log(f"保存配置文件时出错: {e}")

    def setup_history_size_validation(self):
        """设置历史帧数量输入的验证"""

        def validate_history_size(*args):
            # 获取当前值
            current_value = self.history_size_var.get()

            # 如果不是数字，恢复为默认值20
            if not current_value.isdigit():
                self.history_size_var.set("20")
                return

            # 转换为整数
            try:
                history_size = int(current_value)

                # 限制范围在1-100之间
                if history_size < 1:
                    self.history_size_var.set("1")
                elif history_size > 100:
                    self.history_size_var.set("100")
            except ValueError:
                # 如果转换失败，恢复为默认值20
                self.history_size_var.set("20")

        # 添加trace监听变量变化
        self.history_size_var.trace('w', lambda *args: validate_history_size())

    def setup_styles(self):
        """设置UI样式"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabel', background=self.bg_color, font=('Segoe UI', 9))
        style.configure('TButton', font=('Segoe UI', 9), padding=5)
        style.configure('Accent.TButton', background=self.accent_color,
                        foreground='white', font=('Segoe UI', 9, 'bold'))
        style.configure('TProgressbar', thickness=15, background=self.accent_color)
        style.configure('TLabelframe', background=self.bg_color, borderwidth=2)
        style.configure('TLabelframe.Label', background=self.bg_color,
                        font=('Segoe UI', 10, 'bold'))

    def init_history_cache(self):
        """初始化历史帧缓存"""
        # 检查历史帧开关
        if not self.enable_history_var.get():
            # 如果历史帧功能关闭，使用默认值1（只与前一帧比较）
            history_size = 1
        else:
            try:
                history_size = int(self.history_size_var.get())
                # 确保历史帧数量在有效范围内
                if history_size < 1:
                    history_size = 1
                    self.history_size_var.set("1")
                elif history_size > 100:
                    history_size = 100
                    self.history_size_var.set("100")
            except:
                history_size = 20  # 默认值
                self.history_size_var.set("20")

        self.frame_history = collections.deque(maxlen=history_size)
        self.frame_hash_history = collections.deque(maxlen=history_size)

        if self.use_ssim_var.get():
            self.frame_thumbnail_history = collections.deque(maxlen=history_size)
        else:
            self.frame_thumbnail_history = None

        self.frame_sr_history = collections.deque(maxlen=history_size)
        self.frame_idx_history = collections.deque(maxlen=history_size)

        # 记录历史帧设置
        if hasattr(self, 'log_text') and self.enable_history_var.get():
            self.log(f"历史帧功能已启用，缓存大小: {history_size} 帧")
        elif hasattr(self, 'log_text') and not self.enable_history_var.get():
            self.log("历史帧功能已禁用，只与前一帧比较")

    def setup_ui(self):
        """设置UI布局"""
        # 主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 标题
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(title_frame, text="APISR 视频超分辨率处理工具",
                               font=('Segoe UI', 18, 'bold'),
                               foreground=self.sidebar_color,
                               background=self.bg_color)
        title_label.pack(side=tk.LEFT)

        version_label = tk.Label(title_frame, text="v1.8",  # 更新版本号
                                 font=('Segoe UI', 9),
                                 foreground='#7f8c8d',
                                 background=self.bg_color)
        version_label.pack(side=tk.RIGHT)

        # 主内容区域 - 使用PanedWindow实现可调整大小的分割
        paned_window = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, sashwidth=8, sashrelief=tk.RAISED)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 左侧参数面板
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, width=500, minsize=400)

        # 右侧日志面板
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, width=600, minsize=400)

        # 设置面板
        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)

        # 进度条区域
        progress_frame = ttk.Frame(main_container)
        progress_frame.pack(fill=tk.X, pady=(0, 5))

        # 进度信息
        progress_info_frame = ttk.Frame(progress_frame)
        progress_info_frame.pack(fill=tk.X, pady=(0, 3))

        self.progress_info = ttk.Label(progress_info_frame, text="准备开始处理",
                                       font=('Segoe UI', 10, 'bold'),
                                       foreground=self.sidebar_color)
        self.progress_info.pack(side=tk.LEFT, anchor=tk.W)

        self.detailed_progress_info = ttk.Label(progress_info_frame, text="",
                                                font=('Segoe UI', 9),
                                                foreground='#7f8c8d')
        self.detailed_progress_info.pack(side=tk.RIGHT, anchor=tk.E)

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, length=600,
                                            style='TProgressbar')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        # 控制按钮区域
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill=tk.X, pady=(5, 5))

        # 左侧按钮组
        left_btn_frame = ttk.Frame(control_frame)
        left_btn_frame.pack(side=tk.LEFT)

        self.process_btn = ModernButton(left_btn_frame, text="▶ 开始处理",
                                        command=self.start_processing, width=12)
        self.process_btn.pack(side=tk.LEFT, padx=2)

        self.pause_btn = ttk.Button(left_btn_frame, text="⏸ 暂停",
                                    command=self.toggle_pause, width=12, state='disabled')
        self.pause_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(left_btn_frame, text="⏹ 停止",
                                   command=self.stop_processing, width=12, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        # 中间统计信息
        center_btn_frame = ttk.Frame(control_frame)
        center_btn_frame.pack(side=tk.LEFT, padx=20)

        self.dup_info = tk.Label(center_btn_frame, text="重复帧: 0",
                                 font=('Segoe UI', 9),
                                 foreground=self.warning_color,
                                 background=self.bg_color)
        self.dup_info.pack()

        # 右侧按钮组
        right_btn_frame = ttk.Frame(control_frame)
        right_btn_frame.pack(side=tk.RIGHT)

        ttk.Button(right_btn_frame, text="📂 打开目录",
                   command=self.open_output_dir, width=12).pack(side=tk.RIGHT, padx=2)

        ttk.Button(right_btn_frame, text="清理临时文件",
                   command=self.cleanup_temp_files, width=12).pack(side=tk.RIGHT, padx=2)

        ttk.Button(right_btn_frame, text="清空日志",
                   command=self.clear_log, width=12).pack(side=tk.RIGHT, padx=2)

        # 新增：保存配置按钮
        ttk.Button(right_btn_frame, text="保存配置",
                   command=self.save_config, width=12).pack(side=tk.RIGHT, padx=2)

        # 底部状态栏
        status_bar = ttk.Frame(main_container, height=20)
        status_bar.pack(fill=tk.X, pady=(5, 0))

        self.status_label = tk.Label(status_bar, text="准备就绪",
                                     font=('Segoe UI', 9),
                                     foreground=self.success_color,
                                     background=self.bg_color)
        self.status_label.pack(side=tk.LEFT, padx=10)

        gpu_info = self.get_gpu_info()
        self.gpu_label = tk.Label(status_bar, text=gpu_info,
                                  font=('Segoe UI', 9),
                                  foreground='#7f8c8d',
                                  background=self.bg_color)
        self.gpu_label.pack(side=tk.RIGHT, padx=10)

    def setup_config_save_bindings(self):
        """设置配置保存的事件绑定"""
        # 为所有重要变量添加trace，当值改变时自动保存配置
        variables_to_trace = [
            (self.model_var, 'w'),
            (self.scale_var, 'w'),
            (self.segment_duration, 'w'),
            (self.downsample_threshold, 'w'),
            (self.batch_size_var, 'w'),
            (self.tile_size_var, 'w'),
            (self.hash_threshold_var, 'w'),
            (self.ssim_threshold_var, 'w'),
            (self.history_size_var, 'w'),
        ]

        for var, mode in variables_to_trace:
            var.trace(mode, lambda *args: self.save_config())

        # 为BooleanVar添加回调
        boolean_vars = [
            self.float16_var,
            self.crop_for_4x_var,
            self.enable_dup_detect_var,
            self.use_ssim_var,
            self.use_hash_var,
            self.test_mode_var,
            self.enable_history_var,
        ]

        for var in boolean_vars:
            var.trace('w', lambda *args: self.save_config())

    def setup_left_panel(self, parent):
        """设置左侧参数面板"""
        # 创建主框架
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 使用grid布局，4行2列
        for i in range(4):
            main_frame.grid_rowconfigure(i, weight=1, pad=2)
        for i in range(2):
            main_frame.grid_columnconfigure(i, weight=1, pad=5)

        row = 0

        # 1. 文件设置部分 - 占用一行两列
        file_frame = ttk.LabelFrame(main_frame, text="文件设置", padding=8)
        file_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
        file_frame.grid_columnconfigure(1, weight=1)

        # 输入文件 - 紧凑布局
        ttk.Label(file_frame, text="输入视频:", font=('Segoe UI', 9)).grid(row=0, column=0, sticky=tk.W, pady=(0, 2))

        input_btn_frame = ttk.Frame(file_frame)
        input_btn_frame.grid(row=0, column=1, sticky=tk.W, pady=(0, 2))

        ttk.Button(input_btn_frame, text="选择视频文件",
                   command=self.select_input_files, width=18).pack(side=tk.LEFT)

        # 创建标签显示选择的视频数量
        self.input_info_label = tk.Label(input_btn_frame, text="未选择视频",
                                         font=('Segoe UI', 8),
                                         foreground='#7f8c8d',
                                         background=self.bg_color)
        self.input_info_label.pack(side=tk.LEFT, padx=(10, 0))

        # 输出目录 - 紧凑布局
        ttk.Label(file_frame, text="输出目录:", font=('Segoe UI', 9)).grid(row=1, column=0, sticky=tk.W, pady=(2, 0))

        output_entry_frame = ttk.Frame(file_frame)
        output_entry_frame.grid(row=1, column=1, sticky=tk.EW, pady=(2, 0))
        output_entry_frame.grid_columnconfigure(0, weight=1)

        output_entry = ttk.Entry(output_entry_frame, textvariable=self.output_dir, font=('Segoe UI', 9))
        output_entry.grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        ttk.Button(output_entry_frame, text="浏览", command=self.select_output_dir, width=8).grid(row=0, column=1)

        row += 1

        # 2. 模型参数部分
        model_frame = ttk.LabelFrame(main_frame, text="模型参数", padding=8)
        model_frame.grid(row=row, column=0, sticky="nsew", padx=2, pady=2)

        # 使用grid布局内部控件
        ttk.Label(model_frame, text="选择模型:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                   values=list(self.models.keys()),
                                   state="readonly", width=12, font=('Segoe UI', 9))
        model_combo.grid(row=0, column=1, sticky=tk.W, pady=2)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)

        ttk.Label(model_frame, text="缩放因子:").grid(row=1, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        self.scale_combo = ttk.Combobox(model_frame, textvariable=self.scale_var,
                                        state="readonly", width=12, font=('Segoe UI', 9))
        self.scale_combo.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(model_frame, text="分段时长(秒):").grid(row=2, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        ttk.Entry(model_frame, textvariable=self.segment_duration,
                  width=12, font=('Segoe UI', 9)).grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(model_frame, text="下采样阈值:").grid(row=3, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        ttk.Entry(model_frame, textvariable=self.downsample_threshold,
                  width=12, font=('Segoe UI', 9)).grid(row=3, column=1, sticky=tk.W, pady=2)

        # 3. 性能设置部分
        perf_frame = ttk.LabelFrame(main_frame, text="性能设置", padding=8)
        perf_frame.grid(row=row, column=1, sticky="nsew", padx=2, pady=2)

        ttk.Label(perf_frame, text="批处理大小:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        ttk.Entry(perf_frame, textvariable=self.batch_size_var,
                  width=10, font=('Segoe UI', 9)).grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(perf_frame, text="瓦片大小:").grid(row=1, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        ttk.Entry(perf_frame, textvariable=self.tile_size_var,
                  width=10, font=('Segoe UI', 9)).grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(perf_frame, text="数据类型:").grid(row=2, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        ttk.Checkbutton(perf_frame, text="FP16加速",
                        variable=self.float16_var).grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(perf_frame, text="边缘处理:").grid(row=3, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        ttk.Checkbutton(perf_frame, text="4倍缩放时裁剪",
                        variable=self.crop_for_4x_var).grid(row=3, column=1, sticky=tk.W, pady=2)

        row += 1

        # 4. 重复帧检测部分 - 占用一行两列
        dup_frame = ttk.LabelFrame(main_frame, text="重复帧检测设置", padding=8)
        dup_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)

        # 启用检测
        ttk.Checkbutton(dup_frame, text="启用重复帧检测",
                        variable=self.enable_dup_detect_var).grid(row=0, column=0, sticky=tk.W, pady=2, columnspan=2)

        # 检测方法
        method_frame = ttk.Frame(dup_frame)
        method_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(method_frame, text="哈希检测",
                        variable=self.use_hash_var, width=10).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(method_frame, text="SSIM检测",
                        variable=self.use_ssim_var, width=10).pack(side=tk.LEFT)

        # 哈希阈值
        ttk.Label(dup_frame, text="哈希阈值:").grid(row=2, column=0, sticky=tk.W, pady=2)
        hash_frame = ttk.Frame(dup_frame)
        hash_frame.grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Entry(hash_frame, textvariable=self.hash_threshold_var,
                  width=8, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(hash_frame, text="(0-10)", foreground='#7f8c8d', font=('Segoe UI', 8)).pack(side=tk.LEFT)

        # SSIM阈值
        ttk.Label(dup_frame, text="SSIM阈值:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ssim_frame = ttk.Frame(dup_frame)
        ssim_frame.grid(row=3, column=1, sticky=tk.W, pady=2)
        ttk.Entry(ssim_frame, textvariable=self.ssim_threshold_var,
                  width=8, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(ssim_frame, text="(0.9-1.0)", foreground='#7f8c8d', font=('Segoe UI', 8)).pack(side=tk.LEFT)

        # 历史帧设置
        ttk.Label(dup_frame, text="历史帧设置:").grid(row=4, column=0, sticky=tk.W, pady=2)
        history_setting_frame = ttk.Frame(dup_frame)
        history_setting_frame.grid(row=4, column=1, sticky=tk.W, pady=2)

        # 历史帧开关
        self.history_check = ttk.Checkbutton(history_setting_frame, text="启用",
                                             variable=self.enable_history_var,
                                             command=self.toggle_history_settings)
        self.history_check.pack(side=tk.LEFT, padx=(0, 10))

        # 历史帧数量输入框
        history_size_frame = ttk.Frame(history_setting_frame)
        history_size_frame.pack(side=tk.LEFT)

        ttk.Label(history_size_frame, text="数量:").pack(side=tk.LEFT, padx=(0, 5))

        # 创建验证函数，确保只能输入整数
        def validate_integer_input(action, value_if_allowed):
            if action == '1':  # 插入操作
                if value_if_allowed == '':
                    return True
                try:
                    int(value_if_allowed)
                    return True
                except ValueError:
                    return False
            return True

        # 创建历史帧数量输入框
        vcmd = (self.root.register(validate_integer_input), '%d', '%P')
        self.history_entry = ttk.Entry(history_size_frame, textvariable=self.history_size_var,
                                       width=6, font=('Segoe UI', 9),
                                       validate='key', validatecommand=vcmd,
                                       state='normal' if self.enable_history_var.get() else 'disabled')
        self.history_entry.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Label(history_size_frame, text="(1-100)", foreground='#7f8c8d', font=('Segoe UI', 8)).pack(side=tk.LEFT)

        row += 1

        # 5. 其他选项和说明信息部分
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(2, 0))
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)

        # 处理选项部分
        options_frame = ttk.LabelFrame(bottom_frame, text="处理选项", padding=8)
        options_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        ttk.Checkbutton(options_frame, text="测试模式(仅重复帧检测)",
                        variable=self.test_mode_var).pack(anchor=tk.W, pady=2)

        ttk.Checkbutton(options_frame, text="启用配置自动保存",
                        command=self.save_config).pack(anchor=tk.W, pady=2)

        # 说明信息部分
        info_frame = ttk.LabelFrame(bottom_frame, text="说明", padding=8)
        info_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        info_text = """1. 支持批量处理多个视频
2. 暂停时可保存进度
3. 重复帧检测可加速处理
4. 历史帧数量可配置
5. 配置自动保存"""

        info_label = tk.Label(info_frame, text=info_text,
                              font=('Segoe UI', 8),
                              foreground='#7f8c8d',
                              background=self.bg_color,
                              justify=tk.LEFT)
        info_label.pack(anchor=tk.W)

    def setup_right_panel(self, parent):
        """设置右侧日志面板"""
        log_frame = ttk.LabelFrame(parent, text="处理日志", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = ScrolledText(log_frame, height=28, width=60,
                                     font=('Consolas', 9),
                                     bg='#2c3e50', fg='white',
                                     insertbackground='white')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def toggle_history_settings(self):
        """切换历史帧设置的状态"""
        if self.enable_history_var.get():
            self.history_entry.config(state='normal')
            self.log("历史帧功能已启用")
        else:
            self.history_entry.config(state='disabled')
            self.log("历史帧功能已禁用")

    def get_gpu_info(self):
        """获取GPU信息"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return f"GPU: {gpu_name} ({gpu_memory:.1f}GB)"
        return "无可用GPU"

    def on_model_change(self, event=None):
        """当模型改变时更新可用缩放因子"""
        model = self.model_var.get()
        if model in self.models:
            scales = self.models[model]["scale"]
            self.scale_combo['values'] = scales
            if str(scales[0]) in self.scale_var.get():
                self.scale_var.set(str(scales[0]))
            else:
                self.scale_var.set(str(scales[0]))

    def select_input_files(self):
        """选择多个输入视频文件"""
        filenames = filedialog.askopenfilenames(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("所有文件", "*.*")
            ]
        )

        if filenames:
            # 按文件名排序
            self.input_paths = sorted(list(filenames))

            # 更新显示信息
            if len(self.input_paths) == 1:
                file_name = os.path.basename(self.input_paths[0])
                if len(file_name) > 20:
                    file_name = file_name[:17] + "..."
                self.input_info_label.config(text=f"已选择: {file_name}")
            else:
                self.input_info_label.config(text=f"已选择 {len(self.input_paths)} 个视频")
                self.log(f"已选择 {len(self.input_paths)} 个视频文件，将按以下顺序处理:")
                for i, path in enumerate(self.input_paths):
                    self.log(f"  {i + 1}. {os.path.basename(path)}")

            # 自动设置输出目录（以第一个视频的目录为准）
            if not self.output_dir.get():
                output_path = Path(self.input_paths[0]).parent / "APISR_Output"
                self.output_dir.set(str(output_path))

    def select_output_dir(self):
        """选择输出目录"""
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_dir.set(directory)

    def open_output_dir(self):
        """打开输出目录"""
        output_dir = self.output_dir.get()
        if output_dir and os.path.exists(output_dir):
            try:
                if sys.platform == "win32":
                    os.startfile(output_dir)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", output_dir])
                else:
                    subprocess.Popen(["xdg-open", output_dir])
            except Exception as e:
                self.log(f"打开目录失败: {e}")

    def setup_temp_dirs(self, video_path):
        """设置临时目录结构 - 基于视频文件名"""
        output_dir = self.output_dir.get()
        if not output_dir:
            return None

        # 获取视频基础名称
        video_name = Path(video_path).stem

        # 基于视频文件名创建临时目录
        temp_dir_name = f"{video_name}_temp"
        self.temp_base_dir = os.path.join(output_dir, temp_dir_name)

        # 创建标准化的目录结构
        dirs = {
            'base': self.temp_base_dir,
            'original_segments': os.path.join(self.temp_base_dir, "01_original_segments"),
            'audio': os.path.join(self.temp_base_dir, "02_audio"),
            'segment_frames': os.path.join(self.temp_base_dir, "03_segment_frames"),  # 直接放置before/after文件夹
            'processed_segments': os.path.join(self.temp_base_dir, "04_processed_segments"),
            'logs': os.path.join(self.temp_base_dir, "05_logs")
        }

        # 创建目录
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)

        return dirs

    def setup_segment_frame_dirs(self, segment_index):
        """为当前片段设置帧目录 - 直接在03_segment_frames下创建文件夹"""
        if not self.temp_base_dir:
            return None, None

        # 直接在03_segment_frames下创建带前后缀的文件夹
        before_dir = os.path.join(self.temp_base_dir, "03_segment_frames", f"segment_{segment_index:03d}_before")
        after_dir = os.path.join(self.temp_base_dir, "03_segment_frames", f"segment_{segment_index:03d}_after")

        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)

        return before_dir, after_dir

    def cleanup_segment_frame_dirs(self, segment_index):
        """清理当前片段的帧目录"""
        if not self.temp_base_dir:
            return

        # 清理before和after文件夹
        before_dir = os.path.join(self.temp_base_dir, "03_segment_frames", f"segment_{segment_index:03d}_before")
        after_dir = os.path.join(self.temp_base_dir, "03_segment_frames", f"segment_{segment_index:03d}_after")

        for dir_path in [before_dir, after_dir]:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    self.log(f"已清理临时帧目录: {os.path.basename(dir_path)}")
                except Exception as e:
                    self.log(f"清理临时帧目录时出错: {e}")

    def cleanup_temp_files(self):
        """清理临时文件"""
        output_dir = self.output_dir.get()
        if output_dir:
            # 查找所有基于视频文件名的临时目录
            temp_dirs = []
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path) and item.endswith("_temp"):
                    temp_dirs.append(item_path)

            if temp_dirs:
                response = messagebox.askyesno("清理临时文件",
                                               f"找到 {len(temp_dirs)} 个临时目录。是否清理所有临时文件？")
                if response:
                    for temp_dir in temp_dirs:
                        try:
                            shutil.rmtree(temp_dir)
                            self.log(f"已清理临时目录: {os.path.basename(temp_dir)}")
                        except Exception as e:
                            self.log(f"清理临时目录时出错: {e}")
                    messagebox.showinfo("清理完成", "临时文件清理完成")
            else:
                messagebox.showinfo("清理临时文件", "没有找到临时文件")

    def log(self, message):
        """添加日志"""
        if not hasattr(self, 'log_text'):
            # 如果log_text还不存在，先打印到控制台
            print(f"[初始化] {message}")
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)

    def update_status(self, message, color="black"):
        """更新状态"""
        colors = {
            "black": "#2c3e50",
            "green": self.success_color,
            "blue": self.accent_color,
            "orange": self.warning_color,
            "red": self.danger_color
        }
        self.status_label.config(text=message, foreground=colors.get(color, color))
        self.root.update_idletasks()

    def update_progress_info(self):
        """更新进度信息"""
        if self.total_segments > 0:
            info = f"视频 {self.current_video_index + 1}/{len(self.input_paths)} - 片段 {self.current_segment_index + 1}/{self.total_segments}"
            self.progress_info.config(text=info)
            self.root.update_idletasks()

    def update_detailed_progress(self, current_frame, total_frames):
        """更新详细进度信息"""
        if total_frames > 0:
            percentage = current_frame / total_frames * 100
            info = f"当前片段: 第 {current_frame}/{total_frames} 帧 ({percentage:.1f}%)"
            self.detailed_progress_info.config(text=info)
            self.root.update_idletasks()

    def update_dup_info(self, dup_count):
        """更新重复帧信息"""
        self.dup_info.config(text=f"重复帧: {dup_count}")
        self.root.update_idletasks()

    def update_progress(self, value):
        """更新进度条"""
        self.progress_var.set(value)
        self.root.update_idletasks()

    def save_progress(self, force=False):
        """保存进度 - 优化保存频率"""
        if not self.output_dir.get() or not self.temp_base_dir:
            return

        # 检查是否需要保存
        current_time = time.time()
        if not force and current_time - self.last_save_time < self.save_interval:
            return

        # 保存当前视频的进度
        current_video_path = self.input_paths[self.current_video_index] if self.current_video_index < len(
            self.input_paths) else ""

        progress_data = {
            'current_video_index': self.current_video_index,
            'current_video_path': current_video_path,
            'model': self.model_var.get(),
            'scale': int(self.scale_var.get()),
            'downsample_threshold': int(self.downsample_threshold.get()),
            'float16': self.float16_var.get(),
            'crop_for_4x': self.crop_for_4x_var.get(),
            'batch_size': int(self.batch_size_var.get()),
            'hash_threshold': int(self.hash_threshold_var.get()),
            'ssim_threshold': float(self.ssim_threshold_var.get()),
            'use_hash': self.use_hash_var.get(),
            'use_ssim': self.use_ssim_var.get(),
            'test_mode': self.test_mode_var.get(),
            'enable_history': self.enable_history_var.get(),
            'history_size': int(self.history_size_var.get()),
            'current_segment_index': self.current_segment_index,
            'current_frame_in_segment': self.current_frame_in_segment,
            'total_segments': self.total_segments,
            'segments': self.segments,
            'processed_segments': self.processed_segments,
            'temp_base_dir': self.temp_base_dir,
            'dup_frame_count': self.dup_frame_count,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        progress_file = os.path.join(self.temp_base_dir, "progress_data.pkl")
        try:
            with open(progress_file, 'wb') as f:
                pickle.dump(progress_data, f)

            self.last_save_time = current_time
            if force:
                self.log(
                    f"进度已保存: 视频 {self.current_video_index + 1} - 片段 {self.current_segment_index + 1} 的第 {self.current_frame_in_segment + 1} 帧")
        except Exception as e:
            self.log(f"保存进度时出错: {e}")

    # ============================================================
    # 模型加载函数（从test_utils.py整合）
    # ============================================================

    def load_rrdb(self, generator_weight_PATH, scale, print_options=False):
        '''加载RRDB模型'''
        # 加载检查点
        checkpoint_g = torch.load(generator_weight_PATH)

        # 查找生成器权重
        if 'params_ema' in checkpoint_g:
            # 对于官方的ESRNET/ESRGAN权重
            weight = checkpoint_g['params_ema']
            generator = RRDBNet(3, 3, scale=scale)  # 默认块数为6

        elif 'params' in checkpoint_g:
            # 对于官方的ESRNET/ESRGAN权重
            weight = checkpoint_g['params']
            generator = RRDBNet(3, 3, scale=scale)

        elif 'model_state_dict' in checkpoint_g:
            # 对于个人训练的权重
            weight = checkpoint_g['model_state_dict']
            generator = RRDBNet(3, 3, scale=scale)

        else:
            raise ValueError("This weight is not supported")

        # 处理torch.compile权重键重命名
        old_keys = [key for key in weight]
        for old_key in old_keys:
            if old_key[:10] == "_orig_mod.":
                new_key = old_key[10:]
                weight[new_key] = weight[old_key]
                del weight[old_key]

        generator.load_state_dict(weight)
        generator = generator.eval().cuda()

        # 打印选项以显示使用了哪些设置
        if print_options:
            if 'opt' in checkpoint_g:
                for key in checkpoint_g['opt']:
                    value = checkpoint_g['opt'][key]
                    print(f'{key} : {value}')

        return generator

    def load_cunet(self, generator_weight_PATH, scale, print_options=False):
        '''加载CUNET模型'''
        if scale != 2:
            raise NotImplementedError("We only support 2x in CUNET")

        # 加载检查点
        checkpoint_g = torch.load(generator_weight_PATH)

        # 查找生成器权重
        if 'model_state_dict' in checkpoint_g:
            # 对于个人训练的权重
            weight = checkpoint_g['model_state_dict']
            loss = checkpoint_g["lowest_generator_weight"]
            if "iteration" in checkpoint_g:
                iteration = checkpoint_g["iteration"]
            else:
                iteration = "NAN"
            generator = UNet_Full()
            # generator = torch.compile(generator)  # torch.compile
            self.log(f"the generator weight is {loss} at iteration {iteration}")

        else:
            raise ValueError("This weight is not supported")

        # 处理torch.compile权重键重命名
        old_keys = [key for key in weight]
        for old_key in old_keys:
            if old_key[:10] == "_orig_mod.":
                new_key = old_key[10:]
                weight[new_key] = weight[old_key]
                del weight[old_key]

        generator.load_state_dict(weight)
        generator = generator.eval().cuda()

        # 打印选项以显示使用了哪些设置
        if print_options:
            if 'opt' in checkpoint_g:
                for key in checkpoint_g['opt']:
                    value = checkpoint_g['opt'][key]
                    print(f'{key} : {value}')

        return generator

    def load_grl(self, generator_weight_PATH, scale=4):
        '''加载GRL模型'''
        # 加载检查点
        checkpoint_g = torch.load(generator_weight_PATH)

        # 查找生成器权重
        if 'model_state_dict' in checkpoint_g:
            weight = checkpoint_g['model_state_dict']

            # GRL tiny模型（注意：tiny2版本）
            generator = GRL(
                upscale=scale,
                img_size=64,
                window_size=8,
                depths=[4, 4, 4, 4],
                embed_dim=64,
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                mlp_ratio=2,
                qkv_proj_type="linear",
                anchor_proj_type="avgpool",
                anchor_window_down_factor=2,
                out_proj_type="linear",
                conv_type="1conv",
                upsampler="nearest+conv",  # 更改
            ).cuda()

        else:
            raise ValueError("This weight is not supported")

        generator.load_state_dict(weight)
        generator = generator.eval().cuda()

        # 计算参数数量
        num_params = 0
        for p in generator.parameters():
            if p.requires_grad:
                num_params += p.numel()
        self.log(f"GRL模型参数数量: {num_params / 10 ** 6: 0.2f}M")

        return generator

    def load_dat(self, generator_weight_PATH, scale=4):
        '''加载DAT模型'''
        # 加载检查点
        checkpoint_g = torch.load(generator_weight_PATH)

        # 查找生成器权重
        if 'model_state_dict' in checkpoint_g:
            weight = checkpoint_g['model_state_dict']

            # 默认的DAT小模型
            generator = DAT(upscale=4,
                            in_chans=3,
                            img_size=64,
                            img_range=1.,
                            depth=[6, 6, 6, 6, 6, 6],
                            embed_dim=180,
                            num_heads=[6, 6, 6, 6, 6, 6],
                            expansion_factor=2,
                            resi_connection='1conv',
                            split_size=[8, 16],
                            upsampler='pixelshuffledirect',
                            ).cuda()

        else:
            raise ValueError("This weight is not supported")

        generator.load_state_dict(weight)
        generator = generator.eval().cuda()

        # 计算参数数量
        num_params = 0
        for p in generator.parameters():
            if p.requires_grad:
                num_params += p.numel()
        self.log(f"DAT模型参数数量: {num_params / 10 ** 6: 0.2f}M")

        return generator

    # ============================================================
    # 重复帧检测函数
    # ============================================================

    def calculate_frame_hash(self, frame):
        """计算帧的感知哈希值（优化版）"""
        # 先缩小图像以减少计算量
        h, w = frame.shape[:2]
        if h > 360 or w > 480:
            new_h = 360
            new_w = int(w * (360 / h))
            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # 使用更高效的哈希方法
        frame_hash = imagehash.phash(pil_img, hash_size=8)  # 减小哈希大小以提高计算速度

        return frame_hash

    def calculate_ssim_fast(self, frame1, frame2):
        """快速计算SSIM（优化版）"""
        # 将图像缩小以加速计算
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        # 使用较小的固定尺寸
        target_size = (180, 320)  # 16:9的比例

        # 如果图像比目标尺寸大，则缩小
        if h1 > target_size[0] or w1 > target_size[1]:
            scale_factor = min(target_size[0] / h1, target_size[1] / w1)
            new_h = int(h1 * scale_factor)
            new_w = int(w1 * scale_factor)
            gray1 = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (new_w, new_h))
        else:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        if h2 > target_size[0] or w2 > target_size[1]:
            scale_factor = min(target_size[0] / h2, target_size[1] / w2)
            new_h = int(h2 * scale_factor)
            new_w = int(w2 * scale_factor)
            gray2 = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (new_w, new_h))
        else:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        try:
            ssim_value, _ = ssim(gray1, gray2, full=True, data_range=255)
            return ssim_value
        except:
            return 0.0

    def check_frame_duplicate_enhanced(self, frame, frame_idx):
        """增强版重复帧检测，检查最近N帧"""
        if not self.enable_dup_detect_var.get() or not self.frame_history:
            return False, None, None, None

        current_hash = None
        current_thumbnail = None

        # 计算当前帧的信息（按需计算）
        if self.use_hash_var.get():
            current_hash = self.calculate_frame_hash(frame)

        if self.use_ssim_var.get():
            # 保存缩略图用于SSIM计算
            h, w = frame.shape[:2]
            if h > 180 or w > 320:
                new_h = 180
                new_w = int(w * (180 / h))
                current_thumbnail = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                current_thumbnail = frame.copy()

        # 获取阈值
        hash_threshold = int(self.hash_threshold_var.get())
        ssim_threshold = float(self.ssim_threshold_var.get())

        # 从最近帧开始检查（时间上越接近越可能重复）
        best_match_idx = -1
        best_match_reason = ""
        best_hash_diff = None
        best_ssim_value = None

        # 遍历历史帧（从最近的开始）
        for i, (hist_frame, hist_hash, hist_thumbnail, hist_sr_result, hist_frame_idx) in enumerate(
                zip(self.frame_history, self.frame_hash_history,
                    self.frame_thumbnail_history if self.use_ssim_var.get() else [None] * len(self.frame_history),
                    self.frame_sr_history,
                    self.frame_idx_history)):

            # 跳过无效记录
            if hist_frame is None or hist_sr_result is None:
                continue

            # 如果使用哈希检测
            if self.use_hash_var.get() and current_hash is not None and hist_hash is not None:
                hash_diff = current_hash - hist_hash
                if hash_diff <= hash_threshold:
                    # 如果同时启用了SSIM检测，需要验证SSIM
                    if self.use_ssim_var.get():
                        ssim_value = self.calculate_ssim_fast(frame, hist_frame)
                        if ssim_value >= ssim_threshold:
                            best_match_idx = i
                            best_match_reason = f"哈希({hash_diff})和SSIM({ssim_value:.3f})匹配"
                            best_hash_diff = hash_diff
                            best_ssim_value = ssim_value
                            break
                    else:
                        # 只使用哈希检测
                        best_match_idx = i
                        best_match_reason = f"哈希匹配(差异:{hash_diff})"
                        best_hash_diff = hash_diff
                        break
            # 如果只使用SSIM检测
            elif self.use_ssim_var.get() and current_thumbnail is not None and hist_thumbnail is not None:
                ssim_value = self.calculate_ssim_fast(frame, hist_frame)
                if ssim_value >= ssim_threshold:
                    best_match_idx = i
                    best_match_reason = f"SSIM匹配({ssim_value:.3f})"
                    best_ssim_value = ssim_value
                    break

        if best_match_idx >= 0:
            # 找到匹配的帧
            matched_sr_result = self.frame_sr_history[best_match_idx]
            matched_frame_idx = self.frame_idx_history[best_match_idx]

            # 详细日志输出
            history_size = len(self.frame_history)
            log_message = f"帧 {frame_idx:04d}: 与历史帧 {matched_frame_idx:04d} 匹配"
            if best_hash_diff is not None:
                log_message += f", 哈希差异: {best_hash_diff}"
            if best_ssim_value is not None:
                log_message += f", SSIM: {best_ssim_value:.4f}"
            log_message += f", 历史缓存: {history_size} 帧"
            self.log(log_message)

            self.dup_frame_count += 1
            self.update_dup_info(self.dup_frame_count)

            # 更新历史帧信息（将匹配帧移到最近位置）
            if best_match_idx > 0:  # 如果不是已经在最前面
                # 重新排列历史记录，将匹配帧移到最近位置
                items_to_move = [
                    self.frame_history[best_match_idx],
                    self.frame_hash_history[best_match_idx],
                    self.frame_thumbnail_history[best_match_idx] if self.use_ssim_var.get() else None,
                    self.frame_sr_history[best_match_idx],
                    self.frame_idx_history[best_match_idx]
                ]

                # 移除匹配帧
                del self.frame_history[best_match_idx]
                del self.frame_hash_history[best_match_idx]
                if self.use_ssim_var.get():
                    del self.frame_thumbnail_history[best_match_idx]
                del self.frame_sr_history[best_match_idx]
                del self.frame_idx_history[best_match_idx]

                # 插入到最前面（最近位置）
                self.frame_history.appendleft(items_to_move[0])
                self.frame_hash_history.appendleft(items_to_move[1])
                if self.use_ssim_var.get():
                    self.frame_thumbnail_history.appendleft(items_to_move[2])
                self.frame_sr_history.appendleft(items_to_move[3])
                self.frame_idx_history.appendleft(items_to_move[4])

            return True, matched_sr_result.copy(), current_hash, current_thumbnail

        # 如果没有找到匹配帧，也输出日志
        else:
            history_size = len(self.frame_history)
            log_message = f"帧 {frame_idx:04d}: 未发现重复"
            if self.enable_history_var.get():
                log_message += f", 已检查 {history_size} 个历史帧"
            else:
                log_message += ", 历史帧功能已禁用"
            self.log(log_message)

        return False, None, current_hash, current_thumbnail

    def add_frame_to_history(self, frame, frame_hash, frame_thumbnail, sr_result, frame_idx):
        """添加帧到历史记录"""
        # 添加帧数据
        self.frame_history.append(frame.copy())
        self.frame_hash_history.append(frame_hash)
        if self.use_ssim_var.get():
            self.frame_thumbnail_history.append(frame_thumbnail)

        self.frame_sr_history.append(sr_result.copy() if sr_result is not None else None)
        self.frame_idx_history.append(frame_idx)

    # ============================================================
    # 视频处理函数
    # ============================================================

    def extract_audio(self, video_path, audio_path):
        """提取音频"""
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',
            '-acodec', 'copy',
            '-loglevel', 'quiet',
            audio_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"提取音频失败: {e.stderr}")
            return False

    def split_video_by_keyframes(self, video_path, segment_duration, output_dir):
        """按关键帧分割视频"""
        self.log(f"开始分割视频: {os.path.basename(video_path)}")

        segments = []

        # 创建分段目录
        segment_dir = os.path.join(output_dir, "01_original_segments")
        os.makedirs(segment_dir, exist_ok=True)

        # 使用ffmpeg分割视频
        segment_pattern = os.path.join(segment_dir, "segment_%03d.mp4")

        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-c', 'copy',
            '-map', '0',
            '-segment_time', str(segment_duration),
            '-f', 'segment',
            '-reset_timestamps', '1',
            '-segment_format', 'mp4',
            '-segment_list', os.path.join(segment_dir, "segments_list.txt"),
            segment_pattern
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # 获取生成的分段文件
            for f in sorted(os.listdir(segment_dir)):
                if f.startswith("segment_") and f.endswith(".mp4"):
                    segment_file = os.path.join(segment_dir, f)
                    segments.append(segment_file)
                    self.log(f"创建分段 {len(segments)}: {f}")

        except subprocess.CalledProcessError as e:
            self.log(f"视频分割失败: {e.stderr}")
            return []

        self.log(f"视频分割完成，共{len(segments)}段")
        return segments

    def load_model(self):
        """加载模型（测试模式下不加载）"""
        if self.test_mode_var.get():
            self.log("测试模式：跳过模型加载")
            return None

        model_name = self.model_var.get()
        scale = int(self.scale_var.get())

        # 确定权重路径
        if model_name == "RRDB":
            weight_path = self.models[model_name]["weight"][str(scale)]
        else:
            weight_path = self.models[model_name]["weight"]

        # 检查权重文件是否存在
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")

        self.log(f"加载模型: {model_name}, 缩放: {scale}x")
        self.log(f"权重文件: {weight_path}")

        # 设置数据类型
        if self.float16_var.get():
            torch.backends.cudnn.benchmark = True
            self.weight_dtype = torch.float16
            self.log("使用FP16推理模式（加速）")
        else:
            self.weight_dtype = torch.float32
            self.log("使用FP32推理模式（质量优先）")

        # 加载模型
        if model_name == "GRL":
            generator = self.load_grl(weight_path, scale=scale)
        elif model_name == "DAT":
            generator = self.load_dat(weight_path, scale=scale)
        elif model_name == "RRDB":
            generator = self.load_rrdb(weight_path, scale=scale)
        elif model_name == "CUNET":
            generator = self.load_cunet(weight_path, scale=scale)
        else:
            raise ValueError(f"未知模型: {model_name}")

        generator = generator.to(dtype=self.weight_dtype)
        generator.eval()

        # 移动到GPU
        if torch.cuda.is_available():
            generator = generator.cuda()

        return generator

    def process_single_frame(self, frame):
        """处理单帧图像"""
        if self.test_mode_var.get():
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        from torchvision.transforms import ToTensor

        # 预处理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 下采样（如果需要）
        scale = int(self.scale_var.get())
        downsample_threshold = int(self.downsample_threshold.get())

        h, w, _ = frame_rgb.shape
        short_side = min(h, w)

        original_h, original_w = h, w

        if downsample_threshold != -1 and short_side > downsample_threshold:
            rescale_factor = short_side / downsample_threshold
            new_w = int(w / rescale_factor)
            new_h = int(h / rescale_factor)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 裁剪（如果需要）
        if self.crop_for_4x_var.get() and scale == 4:
            h, w, _ = frame_rgb.shape
            if h % 4 != 0:
                frame_rgb = frame_rgb[:4 * (h // 4), :, :]
            if w % 4 != 0:
                frame_rgb = frame_rgb[:, :4 * (w // 4), :]

        # 转换为tensor并进行推理
        img_tensor = ToTensor()(frame_rgb).unsqueeze(0)

        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        img_tensor = img_tensor.to(dtype=self.weight_dtype)

        with torch.no_grad():
            result = self.generator(img_tensor)

        # 转换为numpy数组
        result_np = result[0].cpu().detach().numpy()
        result_np = np.transpose(result_np, (1, 2, 0))
        result_np = np.clip(result_np * 255.0, 0, 255).astype(np.uint8)

        # 如果需要，缩放回原始大小
        if downsample_threshold != -1 and short_side > downsample_threshold:
            output_h = int(original_h * scale)
            output_w = int(original_w * scale)
            result_np = cv2.resize(result_np, (output_w, output_h), interpolation=cv2.INTER_LINEAR)

        return result_np

    def process_frame_with_enhanced_dup_detect(self, frame, frame_idx):
        """处理单帧，包含增强的重复帧检测"""
        is_duplicate = False

        # 检查是否为重复帧
        is_duplicate, matched_sr_result, current_hash, current_thumbnail = \
            self.check_frame_duplicate_enhanced(frame, frame_idx)

        if is_duplicate and matched_sr_result is not None:
            # 找到重复帧，直接使用历史超分辨率结果
            result_np = matched_sr_result

            # 更新历史记录（使用匹配的帧信息）
            if current_hash is None and self.use_hash_var.get():
                current_hash = self.calculate_frame_hash(frame)
            if current_thumbnail is None and self.use_ssim_var.get():
                h, w = frame.shape[:2]
                if h > 180 or w > 320:
                    new_h = 180
                    new_w = int(w * (180 / h))
                    current_thumbnail = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    current_thumbnail = frame.copy()

            self.add_frame_to_history(frame, current_hash, current_thumbnail, result_np, frame_idx)

            return result_np, current_hash, current_thumbnail, is_duplicate

        # 非重复帧，进行超分辨率处理
        result_np = self.process_single_frame(frame)

        # 计算当前帧的信息
        if self.use_hash_var.get() and current_hash is None:
            current_hash = self.calculate_frame_hash(frame)
        if self.use_ssim_var.get() and current_thumbnail is None:
            h, w = frame.shape[:2]
            if h > 180 or w > 320:
                new_h = 180
                new_w = int(w * (180 / h))
                current_thumbnail = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                current_thumbnail = frame.copy()

        # 更新历史记录
        self.add_frame_to_history(frame, current_hash, current_thumbnail, result_np, frame_idx)

        return result_np, current_hash, current_thumbnail, is_duplicate

    def process_segment_frames(self, segment_path, segment_index):
        """处理视频片段的所有帧（逐帧处理）"""
        segment_name = os.path.basename(segment_path)
        self.log(f"处理片段 {segment_index}: {segment_name}")

        if self.test_mode_var.get():
            self.log("测试模式：仅进行重复帧检测，不进行超分辨率处理")

        # 初始化历史帧缓存
        self.init_history_cache()

        # 更新重复帧计数（已重置为0）
        self.update_dup_info(self.dup_frame_count)

        # 为当前片段创建帧目录（直接创建在03_segment_frames下）
        before_dir, after_dir = self.setup_segment_frame_dirs(segment_index)

        if not before_dir or not after_dir:
            self.log("错误：无法创建帧目录")
            return None, None

        # 提取音频
        audio_name = segment_name.replace('.mp4', '.aac')
        audio_path = os.path.join(self.temp_base_dir, "02_audio", audio_name)
        has_audio = self.extract_audio(segment_path, audio_path)

        if has_audio:
            self.log("音频提取成功")
        else:
            self.log("视频无音频或音频提取失败")

        # 读取视频
        cap = cv2.VideoCapture(segment_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            self.log(f"警告: 无法获取片段 {segment_path} 的帧数")
            cap.release()
            return None, None

        # 计算输出尺寸
        scale = int(self.scale_var.get())
        downsample_threshold = int(self.downsample_threshold.get())

        short_side = min(height, width)
        if downsample_threshold != -1 and short_side > downsample_threshold:
            rescale_factor = short_side / downsample_threshold
        else:
            rescale_factor = 1

        # 输出尺寸
        output_width = int(width * scale / rescale_factor)
        output_height = int(height * scale / rescale_factor)

        self.log(f"输入尺寸: {width}x{height}, 输出尺寸: {output_width}x{output_height}")

        # 显示检测参数
        if self.enable_dup_detect_var.get():
            methods = []
            if self.use_hash_var.get():
                methods.append(f"哈希(阈值:{self.hash_threshold_var.get()})")
            if self.use_ssim_var.get():
                methods.append(f"SSIM(阈值:{self.ssim_threshold_var.get()})")

            if self.enable_history_var.get():
                history_size = int(self.history_size_var.get())
                self.log(f"重复帧检测: {', '.join(methods)}，历史帧功能: 启用(数量:{history_size})")
            else:
                self.log(f"重复帧检测: {', '.join(methods)}，历史帧功能: 禁用")

        # 如果从进度恢复，跳过已处理的帧
        start_frame = self.current_frame_in_segment
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.log(f"从第 {start_frame + 1} 帧恢复处理")

        frame_idx = start_frame
        frame_files = []

        # 创建重复帧记录文件
        dup_record_path = os.path.join(self.temp_base_dir, "05_logs", f"segment_{segment_index:03d}_duplicates.txt")
        with open(dup_record_path, 'w', encoding='utf-8') as dup_file:
            dup_file.write(f"片段 {segment_index} 重复帧记录\n")
            dup_file.write(f"哈希阈值: {self.hash_threshold_var.get()}\n")
            dup_file.write(f"SSIM阈值: {self.ssim_threshold_var.get()}\n")
            dup_file.write(f"哈希检测: {'启用' if self.use_hash_var.get() else '禁用'}\n")
            dup_file.write(f"SSIM检测: {'启用' if self.use_ssim_var.get() else '禁用'}\n")
            dup_file.write(f"历史帧功能: {'启用' if self.enable_history_var.get() else '禁用'}\n")
            if self.enable_history_var.get():
                history_size = int(self.history_size_var.get())
                dup_file.write(f"历史帧数量: {history_size}\n")
            dup_file.write("=" * 50 + "\n")
            dup_file.write("帧号\t是否重复\t匹配帧号\t匹配原因\n")

        # 初始化帧计数器
        frames_processed = 0
        segment_dup_count = 0  # 当前片段的重复帧数

        while True:
            # 检查是否被停止
            if self.stopped:
                self.log(f"停止处理：片段 {segment_index} 的第 {frame_idx + 1} 帧")
                break

            # 检查是否暂停 - 使用高效等待
            if self.paused:
                self.log(f"处理暂停于片段 {segment_index} 的第 {frame_idx + 1} 帧")
                self.save_progress(force=True)  # 暂停时立即保存进度

                # 释放GPU内存以降低占用
                if not self.test_mode_var.get():
                    torch.cuda.empty_cache()

                # 高效等待，而不是忙等待
                while self.paused and not self.stopped:
                    time.sleep(1.0)  # 使用较长的休眠时间减少CPU占用
                    if self.paused:  # 再次检查，避免错过状态变化
                        # 在等待期间定期释放GPU内存
                        if frame_idx % 10 == 0 and not self.test_mode_var.get():
                            torch.cuda.empty_cache()

                if self.stopped:
                    break

                # 恢复时重新加载模型（如果需要）
                if not self.test_mode_var.get() and self.generator is None:
                    try:
                        self.generator = self.load_model()
                    except Exception as e:
                        self.log(f"恢复时重新加载模型失败: {e}")
                        break

                self.log(f"处理继续于片段 {segment_index} 的第 {frame_idx + 1} 帧")

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break

            # 保存原始帧到before目录
            before_path = os.path.join(before_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(before_path, frame)

            # 使用增强的重复帧检测处理帧
            sr_frame, current_hash, current_thumbnail, is_duplicate = \
                self.process_frame_with_enhanced_dup_detect(frame, frame_idx)

            # 保存处理后的帧到after目录
            after_path = os.path.join(after_dir, f"frame_{frame_idx:06d}.png")
            sr_frame_bgr = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(after_path, sr_frame_bgr)

            # 添加到帧文件列表
            if not self.test_mode_var.get():
                frame_files.append(after_path)

            # 记录重复帧信息
            with open(dup_record_path, 'a', encoding='utf-8') as dup_file:
                if is_duplicate:
                    matched_idx = self.frame_idx_history[0] if self.frame_idx_history else "未知"
                    dup_file.write(f"{frame_idx}\t是\t{matched_idx}\t重复帧，使用历史结果\n")
                    segment_dup_count += 1
                else:
                    dup_file.write(f"{frame_idx}\t否\t-\t正常处理\n")

            # 更新当前帧
            self.current_frame_in_segment = frame_idx + 1
            frames_processed += 1

            # 更新详细进度
            self.update_detailed_progress(self.current_frame_in_segment, total_frames)

            # 每处理10帧保存一次进度
            if frames_processed % 10 == 0:
                self.save_progress(force=True)

            # 更新进度条
            progress = (self.current_frame_in_segment / total_frames) * 100
            self.update_progress(progress)

            frame_idx += 1

            # 每处理50帧清理一次GPU内存
            if frame_idx % 50 == 0 and not self.test_mode_var.get():
                torch.cuda.empty_cache()

        cap.release()

        # 记录重复帧统计
        if self.enable_dup_detect_var.get():
            self.log(f"片段处理完成: {segment_name}，检测到 {segment_dup_count} 个重复帧")
        else:
            self.log(f"片段处理完成: {segment_name}")

        # 清空历史缓存以释放内存
        self.frame_history.clear()
        self.frame_hash_history.clear()
        if hasattr(self, 'frame_thumbnail_history'):
            self.frame_thumbnail_history.clear()
        self.frame_sr_history.clear()
        self.frame_idx_history.clear()

        return frame_files, audio_path

    def frames_to_video(self, frame_files, output_path, fps, width, height, audio_path=None):
        """将帧序列转换为视频"""
        self.log(f"正在生成视频: {output_path}")

        if not frame_files:
            self.log("错误: 没有可用的帧文件")
            return False

        # 创建临时视频文件（无音频）
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        # 按顺序写入所有帧
        for frame_file in sorted(frame_files):
            if os.path.exists(frame_file):
                frame = cv2.imread(frame_file)
                if frame is not None:
                    # 调整帧大小以匹配输出尺寸
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))
                    out.write(frame)

        out.release()

        # 如果有音频，合并音频和视频
        if audio_path and os.path.exists(audio_path):
            self.log("合并音频和视频...")

            # 使用ffmpeg合并
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                output_path
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                self.log("音频视频合并成功")

                # 删除临时文件
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                return True
            except subprocess.CalledProcessError as e:
                self.log(f"音频视频合并失败: {e.stderr}")
                # 如果合并失败，使用临时视频文件作为输出
                if os.path.exists(temp_video_path):
                    shutil.move(temp_video_path, output_path)
                return True
        else:
            # 如果没有音频，直接使用临时视频文件
            if os.path.exists(temp_video_path):
                shutil.move(temp_video_path, output_path)
                return True

        return False

    def concatenate_videos(self, video_list, output_path):
        """拼接视频片段"""
        self.log("开始拼接视频片段...")

        # 创建临时文件列表
        list_file = tempfile.mktemp(suffix=".txt")

        with open(list_file, 'w', encoding='utf-8') as f:
            for video in video_list:
                f.write(f"file '{os.path.abspath(video)}'\n")

        # 使用ffmpeg拼接
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.log(f"视频拼接完成: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"视频拼接失败: {e.stderr}")
            raise
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)

    def process_single_video(self, video_path):
        """处理单个视频"""
        try:
            self.log(
                f"开始处理视频 {self.current_video_index + 1}/{len(self.input_paths)}: {os.path.basename(video_path)}")

            # 设置视频基础名称
            self.video_base_name = Path(video_path).stem

            # 设置临时目录（基于视频文件名）
            temp_dirs = self.setup_temp_dirs(video_path)
            if not temp_dirs:
                raise ValueError("无法创建临时目录")

            self.log(f"临时文件目录: {temp_dirs['base']}")

            if self.test_mode_var.get():
                self.log("测试模式：仅进行重复帧检测，不进行超分辨率处理")

            # 重置重复帧计数（每个视频开始时重置）
            self.dup_frame_count = 0
            self.update_dup_info(self.dup_frame_count)

            # 检查是否有该视频的进度数据
            progress_file = os.path.join(self.temp_base_dir, "progress_data.pkl")

            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'rb') as f:
                        progress_data = pickle.load(f)

                    # 恢复进度
                    self.current_segment_index = progress_data.get('current_segment_index', 0)
                    self.current_frame_in_segment = progress_data.get('current_frame_in_segment', 0)
                    self.total_segments = progress_data.get('total_segments', 0)
                    self.segments = progress_data.get('segments', [])
                    self.processed_segments = progress_data.get('processed_segments', [])
                    # 注意：不恢复dup_frame_count，已重置为0

                    self.log(
                        f"恢复进度: 片段 {self.current_segment_index + 1}/{self.total_segments} 的第 {self.current_frame_in_segment + 1} 帧")
                    self.log(f"重复帧计数已重置")
                except Exception as e:
                    self.log(f"加载进度数据时出错: {e}")
                    self.current_segment_index = 0
                    self.current_frame_in_segment = 0
                    self.processed_segments = []
                    self.dup_frame_count = 0

            # 步骤1: 加载模型
            self.log("步骤1: 加载模型...")
            self.update_progress(0)
            self.generator = self.load_model()
            self.update_progress(5)

            # 步骤2: 分割视频（如果需要）
            if not self.segments or self.current_segment_index == 0:
                self.log("步骤2: 分割视频...")
                segment_duration = float(self.segment_duration.get())
                self.segments = self.split_video_by_keyframes(video_path, segment_duration, temp_dirs['base'])
                self.total_segments = len(self.segments)
                self.update_progress(10)

                if not self.segments:
                    raise ValueError("视频分割失败")

                # 重置进度
                self.current_segment_index = 0
                self.current_frame_in_segment = 0
                self.processed_segments = []
            else:
                self.log(f"步骤2: 使用已有的 {len(self.segments)} 个片段")
                self.update_progress(10)

            # 保存初始进度
            self.save_progress(force=True)

            # 步骤3: 逐帧处理每个片段
            self.log("步骤3: 处理视频片段...")

            all_processed_frames = []
            all_audio_paths = []

            for i in range(self.current_segment_index, len(self.segments)):
                # 检查是否被停止
                if self.stopped:
                    self.log(f"处理被用户停止于片段 {i + 1}")
                    self.save_progress(force=True)
                    break

                segment = self.segments[i]
                segment_name = os.path.basename(segment)

                # 检查是否已经处理过
                if segment_name in self.processed_segments:
                    self.log(f"跳过已处理的片段 {i + 1}/{len(self.segments)}: {segment_name}")
                    self.current_segment_index = i + 1
                    self.current_frame_in_segment = 0
                    continue

                self.log(f"处理片段 {i + 1}/{len(self.segments)}: {segment_name}")
                frame_files, audio_path = self.process_segment_frames(segment, i + 1)

                # 检查是否被停止
                if self.stopped:
                    self.save_progress(force=True)
                    break

                if frame_files and not self.test_mode_var.get():
                    # 生成处理后的片段视频
                    output_segment = os.path.join(temp_dirs['processed_segments'], f"processed_{segment_name}")

                    # 获取视频参数
                    cap = cv2.VideoCapture(segment)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    # 计算输出尺寸
                    scale = int(self.scale_var.get())
                    downsample_threshold = int(self.downsample_threshold.get())
                    short_side = min(height, width)

                    if downsample_threshold != -1 and short_side > downsample_threshold:
                        rescale_factor = short_side / downsample_threshold
                    else:
                        rescale_factor = 1

                    output_width = int(width * scale / rescale_factor)
                    output_height = int(height * scale / rescale_factor)

                    # 将帧转换为视频
                    self.frames_to_video(frame_files, output_segment, fps, output_width, output_height, audio_path)

                    all_processed_frames.extend(frame_files)
                    if audio_path:
                        all_audio_paths.append(audio_path)
                elif self.test_mode_var.get():
                    self.log(f"测试模式：片段 {i + 1} 处理完成，帧文件已保存")

                # 更新进度
                self.current_segment_index = i + 1
                self.current_frame_in_segment = 0
                self.processed_segments.append(segment_name)
                self.update_progress_info()

                # 保存进度
                self.save_progress(force=True)

                # 清理当前片段的帧目录
                self.log(f"清理片段 {i + 1} 的临时帧文件...")
                self.cleanup_segment_frame_dirs(i + 1)

                # 更新总体进度
                overall_progress = 10 + (i + 1) / len(self.segments) * 60
                self.update_progress(overall_progress)

                # 处理完一个片段后清理GPU内存
                if not self.test_mode_var.get():
                    torch.cuda.empty_cache()

            if self.stopped:
                self.log(
                    f"处理已停止，进度已保存于片段 {self.current_segment_index} 的第 {self.current_frame_in_segment} 帧")
                self.save_progress(force=True)
                return False

            # 步骤4: 如果处理了多个片段且不是测试模式，拼接视频
            if not self.test_mode_var.get():
                self.log("步骤4: 拼接处理后的视频片段...")
                processed_segments_paths = []
                for segment_name in self.processed_segments:
                    processed_path = os.path.join(temp_dirs['processed_segments'], f"processed_{segment_name}")
                    if os.path.exists(processed_path):
                        processed_segments_paths.append(processed_path)

                if processed_segments_paths:
                    if len(processed_segments_paths) > 1:
                        output_filename = f"{self.video_base_name}_super_resolved.mp4"
                        final_output = os.path.join(self.output_dir.get(), output_filename)
                        self.concatenate_videos(processed_segments_paths, final_output)
                    else:
                        # 如果只有一个片段，直接复制
                        output_filename = f"{self.video_base_name}_super_resolved.mp4"
                        final_output = os.path.join(self.output_dir.get(), output_filename)
                        shutil.copy2(processed_segments_paths[0], final_output)

                    self.update_progress(95)
                    self.log(f"最终输出文件: {final_output}")
                else:
                    self.log("没有可拼接的片段")
            else:
                self.log("测试模式：跳过视频合成步骤")
                self.update_progress(95)

            # 步骤5: 清理临时文件和进度记录
            self.log("步骤5: 清理临时文件...")

            # 删除进度文件
            if os.path.exists(progress_file):
                os.remove(progress_file)

            self.update_progress(100)

            if self.test_mode_var.get():
                self.log("测试模式完成！")
                self.log(f"重复帧检测统计：总计检测到 {self.dup_frame_count} 个重复帧")
                self.log(f"测试结果保存在: {temp_dirs['base']}")
            else:
                self.log(f"处理完成！输出文件: {self.video_base_name}_super_resolved.mp4")
                # 显示重复帧统计
                if self.enable_dup_detect_var.get():
                    self.log(f"总计检测到 {self.dup_frame_count} 个重复帧，已复用处理结果，加速了处理速度")

            # 重置状态
            self.current_segment_index = 0
            self.current_frame_in_segment = 0
            self.total_segments = 0
            self.segments = []
            self.processed_segments = []
            self.dup_frame_count = 0
            self.update_dup_info(0)

            # 清空历史缓存
            self.frame_history.clear()
            self.frame_hash_history.clear()
            if hasattr(self, 'frame_thumbnail_history'):
                self.frame_thumbnail_history.clear()
            self.frame_sr_history.clear()
            self.frame_idx_history.clear()

            return True

        except Exception as e:
            self.log(f"处理视频失败: {str(e)}")
            # 保存进度以便恢复
            self.save_progress(force=True)
            return False

    def process_videos(self):
        """主处理函数 - 处理多个视频"""
        try:
            # 检查输入
            if not self.input_paths:
                messagebox.showerror("错误", "请选择有效的输入视频文件")
                return

            output_dir = self.output_dir.get()
            if not output_dir:
                messagebox.showerror("错误", "请选择输出目录")
                return

            # 验证参数
            try:
                hash_threshold = int(self.hash_threshold_var.get())
                ssim_threshold = float(self.ssim_threshold_var.get())
                history_size = int(self.history_size_var.get())

                if hash_threshold < 0 or hash_threshold > 10:
                    messagebox.showwarning("警告", "哈希相似度阈值必须在0-10之间")
                    self.hash_threshold_var.set("3")
                    return

                if ssim_threshold < 0.9 or ssim_threshold > 1.0:
                    messagebox.showwarning("警告", "SSIM阈值必须在0.9-1.0之间")
                    self.ssim_threshold_var.set("0.98")
                    return

                if history_size < 1 or history_size > 100:
                    messagebox.showwarning("警告", "历史帧数量必须在1-100之间")
                    self.history_size_var.set("20")
                    return
            except ValueError:
                messagebox.showerror("错误", "参数格式错误")
                return

            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 更新状态
            self.processing = True
            self.paused = False
            self.stopped = False
            self.process_btn.config(state='disabled')
            self.pause_btn.config(state='normal')
            self.stop_btn.config(state='normal')
            self.update_status("批量处理中...", "blue")

            # 处理每个视频
            total_videos = len(self.input_paths)
            for i in range(self.current_video_index, total_videos):
                if self.stopped:
                    break

                self.current_video_index = i
                video_path = self.input_paths[i]

                # 更新进度信息
                self.progress_info.config(text=f"正在处理视频 {i + 1}/{total_videos}: {os.path.basename(video_path)}")
                self.root.update_idletasks()

                # 处理单个视频
                success = self.process_single_video(video_path)

                if not success and not self.stopped:
                    # 单个视频处理失败，但用户没有停止，继续处理下一个
                    self.log(f"视频处理失败，继续处理下一个视频")
                    continue

                if self.stopped:
                    break

            if self.stopped:
                self.log(f"批量处理已停止，进度已保存")
                self.update_status("已停止，进度已保存", "orange")
                return

            # 所有视频处理完成
            if self.test_mode_var.get():
                self.update_status("测试完成！", "green")
                messagebox.showinfo("测试完成",
                                    f"批量测试完成！\n\n"
                                    f"共处理 {total_videos} 个视频\n"
                                    f"测试结果保存在各视频的临时目录中")
            else:
                self.update_status("批量处理完成！", "green")
                messagebox.showinfo("完成",
                                    f"批量处理完成！\n\n"
                                    f"共处理 {total_videos} 个视频\n"
                                    f"输出目录: {output_dir}")

            # 重置状态
            self.current_video_index = 0

        except Exception as e:
            self.log(f"批量处理失败: {str(e)}")
            self.update_status(f"处理失败: {str(e)}", "red")
            messagebox.showerror("错误", f"处理失败: {str(e)}")
            # 保存进度以便恢复
            self.save_progress(force=True)
        finally:
            self.processing = False
            self.paused = False
            self.stopped = False
            self.process_btn.config(state='normal')
            self.pause_btn.config(state='disabled', text="⏸ 暂停")
            self.stop_btn.config(state='disabled')

            # 清理GPU内存
            if self.generator and not self.test_mode_var.get():
                del self.generator
                torch.cuda.empty_cache()

            # 保存配置
            self.save_config()

    def start_processing(self):
        """开始处理"""
        if self.processing:
            return

        # 验证参数
        try:
            scale = int(self.scale_var.get())
            model = self.model_var.get()
            batch_size = int(self.batch_size_var.get())
            history_size = int(self.history_size_var.get())

            if model in ["GRL", "DAT"] and scale != 4:
                messagebox.showwarning("警告", f"{model}模型只支持4倍缩放")
                self.scale_var.set("4")
                return

            if scale not in [2, 4]:
                messagebox.showwarning("警告", "缩放因子必须是2或4")
                return

            if batch_size < 1 or batch_size > 2:
                messagebox.showwarning("警告", "6GB GPU批处理大小建议为1-2")
                self.batch_size_var.set("1")
                return

            if history_size < 1 or history_size > 100:
                messagebox.showwarning("警告", "历史帧数量必须在1-100之间")
                self.history_size_var.set("20")
                return
        except ValueError:
            messagebox.showerror("错误", "参数格式错误")
            return

        # 在新线程中运行处理
        self.processing_thread = threading.Thread(target=self.process_videos)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def toggle_pause(self):
        """切换暂停/继续状态"""
        if not self.processing:
            return

        if self.paused:
            # 继续处理
            self.paused = False
            self.pause_btn.config(text="⏸ 暂停")
            self.update_status("处理中...", "blue")
            self.log("处理继续")

            # 通知暂停的线程继续
            with self.pause_cv:
                self.pause_cv.notify_all()
        else:
            # 暂停处理
            self.paused = True
            self.pause_btn.config(text="▶ 继续")
            self.update_status("已暂停", "orange")
            self.log("处理暂停，保存进度...")
            self.save_progress(force=True)

    def stop_processing(self):
        """停止处理"""
        if not self.processing:
            return

        response = messagebox.askyesnocancel("停止处理",
                                             "请选择停止方式：\n\n"
                                             "是：保存进度并停止，下次可以继续\n"
                                             "否：直接停止，不保存进度\n"
                                             "取消：返回继续处理")

        if response is None:  # 取消
            return

        if response:  # 是：保存进度并停止
            self.log("正在停止处理并保存进度...")
            self.update_status("正在停止并保存进度...", "orange")
            self.stopped = True
            self.paused = False  # 确保暂停状态被清除

            # 通知暂停的线程继续（如果是暂停状态）
            with self.pause_cv:
                self.pause_cv.notify_all()

            # 等待处理线程响应
            time.sleep(0.5)

            # 强制保存进度
            self.save_progress(force=True)

            # 等待处理线程结束
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)

            self.log("处理已停止，进度已保存")
            self.update_status("已停止，进度已保存", "orange")

        else:  # 否：直接停止，不保存进度
            self.log("正在停止处理，不保存进度...")
            self.update_status("正在停止...", "orange")
            self.stopped = True
            self.paused = False  # 确保暂停状态被清除

            # 通知暂停的线程继续（如果是暂停状态）
            with self.pause_cv:
                self.pause_cv.notify_all()

            # 等待处理线程响应
            time.sleep(0.5)

            # 删除进度文件
            if self.temp_base_dir:
                progress_file = os.path.join(self.temp_base_dir, "progress_data.pkl")
                if os.path.exists(progress_file):
                    try:
                        os.remove(progress_file)
                        self.log("已删除进度文件")
                    except:
                        pass

            # 等待处理线程结束
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)

            self.log("处理已停止，进度未保存")
            self.update_status("已停止，进度未保存", "orange")

            # 重置进度
            self.current_video_index = 0
            self.current_segment_index = 0
            self.current_frame_in_segment = 0
            self.dup_frame_count = 0
            self.update_dup_info(0)

        # 重置按钮状态
        self.process_btn.config(state='normal')
        self.pause_btn.config(state='disabled', text="⏸ 暂停")
        self.stop_btn.config(state='disabled')
        self.paused = False
        self.processing = False

    def run(self):
        """运行GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 居中显示窗口
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

        self.root.mainloop()

    def on_closing(self):
        """关闭窗口时的清理"""
        # 保存配置
        self.save_config()

        if self.processing:
            response = messagebox.askyesnocancel("退出",
                                                 "处理仍在进行中，您可以选择:\n\n"
                                                 "是: 保存进度并退出\n"
                                                 "否: 不保存进度直接退出\n"
                                                 "取消: 继续处理")

            if response is True:  # 保存进度并退出
                self.log("保存进度并退出...")
                self.save_progress(force=True)
                self.processing = False
                self.stopped = True
                self.paused = False

                # 通知暂停的线程继续（如果是暂停状态）
                with self.pause_cv:
                    self.pause_cv.notify_all()

                time.sleep(1.0)  # 给线程更多时间响应
                self.root.destroy()
            elif response is False:  # 直接退出
                self.log("直接退出，不保存进度")
                self.processing = False
                self.stopped = True
                self.paused = False

                # 通知暂停的线程继续（如果是暂停状态）
                with self.pause_cv:
                    self.pause_cv.notify_all()

                time.sleep(1.0)  # 给线程更多时间响应
                self.root.destroy()
            # 如果选择取消，什么都不做，继续处理
        else:
            self.root.destroy()


def main():
    """主函数"""
    app = APISRVideoProcessor()
    app.run()


if __name__ == "__main__":
    main()