import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import warnings
from datetime import datetime
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import cv2
import imagehash
import numpy as np
import torch
from PIL import Image
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

from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


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
        self.hash_threshold_var = tk.StringVar(value="3")
        self.ssim_threshold_var = tk.StringVar(value="0.98")
        self.enable_dup_detect_var = tk.BooleanVar(value=True)
        self.use_ssim_var = tk.BooleanVar(value=True)
        self.use_hash_var = tk.BooleanVar(value=True)
        self.test_mode_var = tk.BooleanVar(value=False)
        self.enable_history_var = tk.BooleanVar(value=True)
        self.history_size_var = tk.StringVar(value="20")  # 默认值改为20
        self.immediate_merge_var = tk.BooleanVar(value=False)  # 新增：立即合成视频选项
        self.video_encoder_mode = tk.StringVar(value="auto")  # 新增：视频编码器模式
        self.last_test_mode_state = False  # 记录上一次的测试模式状态

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

        # 重复帧检测相关
        self.dup_frame_count = 0

        # 新增：历史帧缓存系统
        self.init_history_cache()

        # 临时文件路径
        self.temp_base_dir = None
        self.current_segment_frames_dir = None
        self.video_base_name = None
        self.is_test_mode_folder = False  # 新增：标记是否为测试模式文件夹

        # 线程控制
        self.processing_thread = None
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.processing_lock = threading.Lock()

        # 新增：暂停时的内存优化
        self.pause_lock = threading.Lock()
        self.pause_cv = threading.Condition(self.pause_lock)
        self.should_sleep = False

        # 新增：内存监控
        self.monitor_thread = None
        self.memory_check_interval = 30  # 30秒检查一次内存

        # 设置历史帧数量验证
        self.setup_history_size_validation()

        self.setup_ui()

        # 设置初始模型
        self.on_model_change()

        # 加载配置文件
        self.load_config()

        # 绑定配置保存事件
        self.setup_config_save_bindings()

        # 跟踪测试模式变化
        self.test_mode_var.trace('w', self.on_test_mode_changed)

    def on_test_mode_changed(self, *args):
        """测试模式变化时的处理"""
        current_state = self.test_mode_var.get()

        # 如果从非测试模式切换到测试模式
        if current_state and not self.last_test_mode_state:
            # 弹出确认窗口
            response = messagebox.askyesno("确认测试模式",
                                           "测试模式仅进行重复帧检测，不进行超分辨率处理，且会创建单独的测试文件夹。\n\n"
                                           "是否确认启用测试模式？")

            if not response:
                # 用户取消，恢复原来的状态
                self.test_mode_var.set(False)
                return
            else:
                self.log("测试模式已启用 - 仅进行重复帧检测，不进行超分辨率处理")

        # 如果从测试模式切换到非测试模式
        elif not current_state and self.last_test_mode_state:
            response = messagebox.askyesno("退出测试模式",
                                           "退出测试模式将删除测试模式产生的临时文件。\n\n"
                                           "是否确认退出测试模式？")

            if response:
                # 清理测试模式的临时文件
                self.cleanup_test_mode_files()
                self.log("已退出测试模式，测试文件已清理")
            else:
                # 用户取消，恢复测试模式
                self.test_mode_var.set(True)
                return

        # 更新状态记录
        self.last_test_mode_state = current_state

        # 自动保存配置
        self.save_config()

    def cleanup_test_mode_files(self):
        """清理测试模式产生的临时文件"""
        output_dir = self.output_dir.get()
        if not output_dir or not os.path.exists(output_dir):
            return

        # 查找所有测试模式的临时目录
        test_temp_dirs = []
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item.endswith("_test_temp"):
                test_temp_dirs.append(item_path)

        if test_temp_dirs:
            self.log(f"找到 {len(test_temp_dirs)} 个测试模式临时目录")
            for temp_dir in test_temp_dirs:
                try:
                    shutil.rmtree(temp_dir)
                    self.log(f"已清理测试临时目录: {os.path.basename(temp_dir)}")
                except Exception as e:
                    self.log(f"清理测试临时目录时出错: {e}")

    def setup_history_size_validation(self):
        """设置历史帧数量输入的验证 - 修改：移除原来的trace验证，改为焦点离开时调整"""

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

        vcmd = (self.root.register(validate_integer_input), '%d', '%P')

        # 在setup_ui中创建输入框时使用这个验证函数
        self.history_validation_command = vcmd

    def adjust_history_size(self, event=None):
        """调整历史帧数量为最接近的10的倍数"""
        try:
            current_value = self.history_size_var.get()

            # 如果为空，设置为默认值20
            if not current_value:
                self.history_size_var.set("20")
                return

            # 转换为整数
            history_size = int(current_value)

            # 限制范围在1-200之间
            if history_size < 1:
                history_size = 10  # 最小值设为10
            elif history_size > 200:
                history_size = 200

            # 调整为最接近的10的倍数
            history_size = round(history_size / 10) * 10

            # 确保调整后仍在有效范围内
            if history_size < 10:
                history_size = 10
            elif history_size > 200:
                history_size = 200

            # 更新变量
            new_value = str(history_size)
            if new_value != current_value:
                self.history_size_var.set(new_value)
                self.log(f"历史帧数量已调整为: {new_value} (最接近的10的倍数)")

        except ValueError:
            # 如果转换失败，设为默认值20
            self.history_size_var.set("20")

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
        """初始化历史帧缓存 - 修复版本"""
        # 检查历史帧开关
        if not self.enable_history_var.get():
            # 如果历史帧功能关闭，使用默认值1（只与前一帧比较）
            history_size = 1
        else:
            try:
                history_size = int(self.history_size_var.get())
                # 确保历史帧数量在有效范围内
                if history_size < 1:
                    history_size = 10
                    self.history_size_var.set("10")
                elif history_size > 200:  # 上限改为200
                    history_size = 200
                    self.history_size_var.set("200")
            except:
                history_size = 20  # 默认值
                self.history_size_var.set("20")

        # 确保deque有最大长度限制
        self.frame_history = collections.deque(maxlen=history_size)
        self.frame_hash_history = collections.deque(maxlen=history_size)

        if self.use_ssim_var.get():
            self.frame_thumbnail_history = collections.deque(maxlen=history_size)
        else:
            self.frame_thumbnail_history = None

        self.frame_sr_history = collections.deque(maxlen=history_size)
        self.frame_idx_history = collections.deque(maxlen=history_size)

    def clear_history_cache(self):
        """清空历史缓存"""
        if hasattr(self, 'frame_history'):
            self.frame_history.clear()
        if hasattr(self, 'frame_hash_history'):
            self.frame_hash_history.clear()
        if hasattr(self, 'frame_thumbnail_history') and self.frame_thumbnail_history:
            self.frame_thumbnail_history.clear()
        if hasattr(self, 'frame_sr_history'):
            self.frame_sr_history.clear()
        if hasattr(self, 'frame_idx_history'):
            self.frame_idx_history.clear()

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

        version_label = tk.Label(title_frame, text="v2.0",  # 版本更新到2.0
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
        """设置配置自动保存的事件绑定"""
        # 为所有重要变量添加trace，当值改变时自动保存配置
        variables_to_trace = [
            (self.model_var, 'w'),
            (self.scale_var, 'w'),
            (self.segment_duration, 'w'),
            (self.downsample_threshold, 'w'),
            (self.hash_threshold_var, 'w'),
            (self.ssim_threshold_var, 'w'),
            (self.history_size_var, 'w'),
            (self.video_encoder_mode, 'w'),  # 新增：视频编码器模式
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
            self.immediate_merge_var,
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

        # 3. 性能设置部分 - 修改：简化了内容
        perf_frame = ttk.LabelFrame(main_frame, text="性能设置", padding=8)
        perf_frame.grid(row=row, column=1, sticky="nsew", padx=2, pady=2)

        ttk.Label(perf_frame, text="数据类型:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        ttk.Checkbutton(perf_frame, text="FP16加速",
                        variable=self.float16_var).grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(perf_frame, text="视频编码:").grid(row=1, column=0, sticky=tk.W, pady=2, padx=(0, 5))
        encoder_combo = ttk.Combobox(perf_frame, textvariable=self.video_encoder_mode,
                                     values=["auto", "opencv", "ffmpeg"], width=10, state="readonly")
        encoder_combo.grid(row=1, column=1, sticky=tk.W, pady=2)

        # 添加两行空行以保持布局平衡
        ttk.Label(perf_frame, text="").grid(row=2, column=0, pady=2)
        ttk.Label(perf_frame, text="").grid(row=3, column=0, pady=2)

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

        # 创建历史帧数量输入框 - 修改：使用验证命令并绑定焦点离开事件
        self.history_entry = ttk.Entry(history_size_frame, textvariable=self.history_size_var,
                                       width=6, font=('Segoe UI', 9),
                                       validate='key', validatecommand=self.history_validation_command,
                                       state='normal' if self.enable_history_var.get() else 'disabled')
        self.history_entry.pack(side=tk.LEFT, padx=(0, 5))

        # 绑定焦点离开事件
        self.history_entry.bind('<FocusOut>', self.adjust_history_size)

        ttk.Label(history_size_frame, text="(1-200)", foreground='#7f8c8d', font=('Segoe UI', 8)).pack(side=tk.LEFT)

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

        # 新增：立即合成视频选项
        ttk.Checkbutton(options_frame, text="立即合成视频",
                        variable=self.immediate_merge_var).pack(anchor=tk.W, pady=2)

        # 说明信息部分
        info_frame = ttk.LabelFrame(bottom_frame, text="说明", padding=8)
        info_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        info_text = """1. 支持批量处理多个视频
2. 哈希阈值越小，SSIM阈值越大，重复检测越严格
3. 取消重复帧识别会切换直接处理视频模式，无法暂停
4. 可利用测试模式自行调整参数
5. 配置自动保存
6. 进度根据临时文件读取，请不要挪动临时文件
7. 所有视频片段处理完后都会立即合成视频
8. 开启立即合成功能会在每个片段完成后合并到整体视频"""

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
        """设置临时目录结构 - 基于视频文件名和测试模式"""
        output_dir = self.output_dir.get()
        if not output_dir:
            return None

        # 获取视频基础名称
        video_name = Path(video_path).stem

        # 根据测试模式添加后缀
        if self.test_mode_var.get():
            temp_dir_suffix = "_test_temp"
            self.is_test_mode_folder = True
        else:
            temp_dir_suffix = "_temp"
            self.is_test_mode_folder = False

        # 基于视频文件名创建临时目录
        temp_dir_name = f"{video_name}{temp_dir_suffix}"
        self.temp_base_dir = os.path.join(output_dir, temp_dir_name)

        # 创建标准化的目录结构 - 删除05_logs相关
        dirs = {
            'base': self.temp_base_dir,
            'original_segments': os.path.join(self.temp_base_dir, "01_original_segments"),
            'audio': os.path.join(self.temp_base_dir, "02_audio"),
            'segment_frames': os.path.join(self.temp_base_dir, "03_segment_frames"),  # 直接放置before/after文件夹
            'processed_segments': os.path.join(self.temp_base_dir, "04_processed_segments"),
            'immediate_merge': os.path.join(self.temp_base_dir, "05_immediate_merge")  # 新增：立即合成目录
        }

        # 创建目录
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)

        return dirs

    def setup_segment_frame_dirs(self, segment_path):
        """为当前片段设置帧目录 - 根据01_original_segments里的文件名来命名"""
        if not self.temp_base_dir:
            return None, None

        # 从segment_path中获取文件名（不带扩展名）
        segment_name = Path(segment_path).stem  # 例如：segment_000

        # 直接在03_segment_frames下创建带前后缀的文件夹
        before_dir = os.path.join(self.temp_base_dir, "03_segment_frames", f"{segment_name}_before")
        after_dir = os.path.join(self.temp_base_dir, "03_segment_frames", f"{segment_name}_after")

        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)

        return before_dir, after_dir

    def cleanup_segment_frame_dirs(self, segment_path):
        """清理当前片段的帧目录"""
        if not self.temp_base_dir:
            return

        # 从segment_path中获取文件名（不带扩展名）
        segment_name = Path(segment_path).stem

        # 清理before和after文件夹
        before_dir = os.path.join(self.temp_base_dir, "03_segment_frames", f"{segment_name}_before")
        after_dir = os.path.join(self.temp_base_dir, "03_segment_frames", f"{segment_name}_after")

        for dir_path in [before_dir, after_dir]:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    pass

    def cleanup_temp_files(self):
        """清理临时文件"""
        output_dir = self.output_dir.get()
        if output_dir:
            # 查找所有基于视频文件名的临时目录
            temp_dirs = []
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path) and (item.endswith("_temp") or item.endswith("_test_temp")):
                    temp_dirs.append(item_path)

            if temp_dirs:
                response = messagebox.askyesno("清理临时文件",
                                               f"找到 {len(temp_dirs)} 个临时目录。是否清理所有临时文件？\n"
                                               f"（包括普通模式和测试模式的临时文件）")
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

    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 安全的数值加载函数
                def safe_get_int(key, default, config_dict):
                    value = config_dict.get(key, default)
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        return default

                def safe_get_float(key, default, config_dict):
                    value = config_dict.get(key, default)
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                # 设置变量，使用安全转换
                if 'model' in config:
                    self.model_var.set(config['model'])
                if 'scale' in config:
                    self.scale_var.set(str(safe_get_int('scale', 4, config)))
                if 'segment_duration' in config:
                    self.segment_duration.set(str(safe_get_int('segment_duration', 20, config)))
                if 'downsample_threshold' in config:
                    self.downsample_threshold.set(str(safe_get_int('downsample_threshold', 720, config)))
                if 'float16' in config:
                    self.float16_var.set(config['float16'])
                if 'crop_for_4x' in config:
                    self.crop_for_4x_var.set(config['crop_for_4x'])
                if 'hash_threshold' in config:
                    self.hash_threshold_var.set(str(safe_get_int('hash_threshold', 3, config)))
                if 'ssim_threshold' in config:
                    self.ssim_threshold_var.set(str(safe_get_float('ssim_threshold', 0.98, config)))
                if 'enable_dup_detect' in config:
                    self.enable_dup_detect_var.set(config['enable_dup_detect'])
                if 'use_ssim' in config:
                    self.use_ssim_var.set(config['use_ssim'])
                if 'use_hash' in config:
                    self.use_hash_var.set(config['use_hash'])
                if 'test_mode' in config:
                    self.test_mode_var.set(config['test_mode'])
                    self.last_test_mode_state = config['test_mode']
                if 'enable_history' in config:
                    self.enable_history_var.set(config['enable_history'])
                if 'history_size' in config:
                    self.history_size_var.set(str(safe_get_int('history_size', 20, config)))
                if 'immediate_merge' in config:
                    self.immediate_merge_var.set(config['immediate_merge'])
                if 'video_encoder_mode' in config:
                    self.video_encoder_mode.set(config['video_encoder_mode'])

                self.log(f"已从 {self.config_file} 加载配置")

                # 更新UI状态
                self.on_model_change()
                self.toggle_history_settings()

            except Exception as e:
                self.log(f"加载配置文件时出错: {e}")
                import traceback
                self.log(f"错误详情: {traceback.format_exc()}")
        else:
            self.log("未找到配置文件，使用默认配置")

    def save_config(self):
        """保存配置文件（永远自动保存）"""
        try:
            # 使用默认值处理空字符串或无效输入
            def get_int_value(var, default):
                value = var.get()
                try:
                    return int(value) if value else default
                except ValueError:
                    return default

            def get_float_value(var, default):
                value = var.get()
                try:
                    return float(value) if value else default
                except ValueError:
                    return default

            # 获取所有配置值，使用默认值处理空字符串
            config = {
                'model': self.model_var.get(),
                'scale': get_int_value(self.scale_var, 4),
                'segment_duration': get_int_value(self.segment_duration, 20),
                'downsample_threshold': get_int_value(self.downsample_threshold, 720),
                'float16': self.float16_var.get(),
                'crop_for_4x': self.crop_for_4x_var.get(),
                'hash_threshold': get_int_value(self.hash_threshold_var, 3),
                'ssim_threshold': get_float_value(self.ssim_threshold_var, 0.98),
                'enable_dup_detect': self.enable_dup_detect_var.get(),
                'use_ssim': self.use_ssim_var.get(),
                'use_hash': self.use_hash_var.get(),
                'test_mode': self.test_mode_var.get(),
                'enable_history': self.enable_history_var.get(),
                'history_size': get_int_value(self.history_size_var, 20),
                'immediate_merge': self.immediate_merge_var.get(),
                'video_encoder_mode': self.video_encoder_mode.get(),
                'last_saved': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 修正了日期格式错误
            }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)

        except Exception as e:
            self.log(f"保存配置文件时出错: {e}")
            # 打印详细错误信息以帮助调试
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")

    # ============================================================
    # 内存监控和清理函数
    # ============================================================

    def add_memory_monitoring(self):
        """添加内存使用监控功能"""
        try:
            if torch.cuda.is_available():
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.log(f"GPU内存使用: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)")
        except ImportError:
            # 如果GPUtil不可用，跳过
            pass
        except Exception as e:
            # 监控出错时不中断处理
            pass

    def start_memory_monitor(self):
        """启动内存监控线程"""

        def monitor_loop():
            while self.processing:
                time.sleep(self.memory_check_interval)  # 每30秒检查一次
                try:
                    if not self.paused and not self.stopped:
                        self.add_memory_monitoring()
                except:
                    pass

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def cleanup_memory(self):
        """清理内存"""
        try:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 清理Python内存
            import gc
            gc.collect()

            # 清理OpenCV缓冲区（如果有）
            try:
                cv2.destroyAllWindows()
            except:
                pass

        except Exception as e:
            self.log(f"内存清理时出错: {e}")

    # ============================================================
    # 模型加载函数（从test_utils.py整合）
    # ============================================================

    def load_rrdb(self, generator_weight_PATH, scale, print_options=False):
        '''加载RRDB模型'''
        start_time = time.time()

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

        elapsed = time.time() - start_time
        self.log(f"RRDB模型加载耗时: {elapsed:.2f}秒")

        return generator

    def load_cunet(self, generator_weight_PATH, scale, print_options=False):
        '''加载CUNET模型'''
        start_time = time.time()

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

        elapsed = time.time() - start_time
        self.log(f"CUNET模型加载耗时: {elapsed:.2f}秒")

        return generator

    def load_grl(self, generator_weight_PATH, scale=4):
        '''加载GRL模型'''
        start_time = time.time()

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

        elapsed = time.time() - start_time
        self.log(f"GRL模型加载耗时: {elapsed:.2f}秒")
        self.log(f"GRL模型参数数量: {num_params / 10 ** 6: 0.2f}M")

        return generator

    def load_dat(self, generator_weight_PATH, scale=4):
        '''加载DAT模型'''
        start_time = time.time()

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

        elapsed = time.time() - start_time
        self.log(f"DAT模型加载耗时: {elapsed:.2f}秒")
        self.log(f"DAT模型参数数量: {num_params / 10 ** 6: 0.2f}M")

        return generator

    # ============================================================
    # 重复帧检测函数
    # ============================================================

    def calculate_frame_hash(self, frame):
        """计算帧的感知哈希值（优化版）"""
        start_time = time.time()

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

        elapsed = time.time() - start_time
        return frame_hash, elapsed

    def calculate_ssim_fast(self, frame1, frame2):
        """快速计算SSIM（优化版）"""
        start_time = time.time()

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
            elapsed = time.time() - start_time
            return ssim_value, elapsed
        except:
            return 0.0, time.time() - start_time

    def check_frame_duplicate_enhanced(self, frame, frame_idx):
        """增强版重复帧检测，检查最近N帧"""
        if not self.enable_dup_detect_var.get() or not self.frame_history:
            return False, None, None, None

        total_start_time = time.time()
        history_size = len(self.frame_history)

        current_hash = None
        current_thumbnail = None

        # 计算当前帧的信息（按需计算）
        hash_time = 0
        if self.use_hash_var.get():
            hash_start = time.time()
            current_hash, hash_time = self.calculate_frame_hash(frame)
            hash_time = time.time() - hash_start

        ssim_thumbnail_time = 0
        if self.use_ssim_var.get():
            # 保存缩略图用于SSIM计算
            thumb_start = time.time()
            h, w = frame.shape[:2]
            if h > 180 or w > 320:
                new_h = 180
                new_w = int(w * (180 / h))
                current_thumbnail = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                current_thumbnail = frame.copy()
            ssim_thumbnail_time = time.time() - thumb_start

        # 获取阈值
        hash_threshold = int(self.hash_threshold_var.get())
        ssim_threshold = float(self.ssim_threshold_var.get())

        # 从最近帧开始检查（时间上越接近越可能重复）
        best_match_idx = -1
        best_match_reason = ""
        best_hash_diff = None
        best_ssim_value = None
        detected_hash_diff = None
        detected_ssim_value = None

        # 遍历历史帧（从最近的开始）
        compare_start = time.time()
        ssim_compare_time = 0
        hash_compare_time = 0

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
                hash_compare_start = time.time()
                hash_diff = current_hash - hist_hash
                hash_compare_time += time.time() - hash_compare_start

                # 记录哈希差值（用于日志输出）
                detected_hash_diff = hash_diff

                if hash_diff <= hash_threshold:
                    # 如果同时启用了SSIM检测，需要验证SSIM
                    if self.use_ssim_var.get():
                        ssim_compare_start = time.time()
                        ssim_value, ssim_elapsed = self.calculate_ssim_fast(frame, hist_frame)
                        ssim_compare_time += ssim_elapsed

                        # 记录SSIM值（用于日志输出）
                        detected_ssim_value = ssim_value

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
                ssim_compare_start = time.time()
                ssim_value, ssim_elapsed = self.calculate_ssim_fast(frame, hist_frame)
                ssim_compare_time += ssim_elapsed

                # 记录SSIM值（用于日志输出）
                detected_ssim_value = ssim_value

                if ssim_value >= ssim_threshold:
                    best_match_idx = i
                    best_match_reason = f"SSIM匹配({ssim_value:.3f})"
                    best_ssim_value = ssim_value
                    break

        compare_time = time.time() - compare_start
        total_elapsed = time.time() - total_start_time

        # 构建检测值字符串
        detection_values = []
        if self.use_hash_var.get() and detected_hash_diff is not None:
            detection_values.append(f"哈希差: {detected_hash_diff}")
        if self.use_ssim_var.get() and detected_ssim_value is not None:
            detection_values.append(f"SSIM: {detected_ssim_value:.3f}")

        detection_str = "，".join(detection_values)

        if best_match_idx >= 0:
            # 找到匹配的帧
            matched_sr_result = self.frame_sr_history[best_match_idx]
            matched_frame_idx = self.frame_idx_history[best_match_idx]

            # 构建详细的时间统计
            time_stats = []
            if hash_time > 0:
                time_stats.append(f"哈希:{hash_time:.3f}s")
            if ssim_thumbnail_time > 0:
                time_stats.append(f"缩略图:{ssim_thumbnail_time:.3f}s")
            if hash_compare_time > 0:
                time_stats.append(f"哈希比较:{hash_compare_time:.3f}s")
            if ssim_compare_time > 0:
                time_stats.append(f"SSIM比较:{ssim_compare_time:.3f}s")

            time_str = "，".join(time_stats) if time_stats else ""

            # 构建日志消息
            log_message = f"帧 {frame_idx:04d}: 与帧 {matched_frame_idx:04d} 重复"
            if detection_str:
                log_message += f" ({detection_str})"
            if time_str:
                log_message += f" [{time_str}]"
            log_message += f" - 总耗时:{total_elapsed:.3f}s"

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

        # 如果没有找到匹配帧
        else:
            # 构建详细的时间统计
            time_stats = []
            if hash_time > 0:
                time_stats.append(f"哈希:{hash_time:.3f}s")
            if ssim_thumbnail_time > 0:
                time_stats.append(f"缩略图:{ssim_thumbnail_time:.3f}s")
            if hash_compare_time > 0:
                time_stats.append(f"哈希比较:{hash_compare_time:.3f}s")
            if ssim_compare_time > 0:
                time_stats.append(f"SSIM比较:{ssim_compare_time:.3f}s")

            time_str = "，".join(time_stats) if time_stats else ""

            # 构建日志消息
            log_message = f"帧 {frame_idx:04d}: 未重复"
            if detection_str:
                log_message += f" ({detection_str})"
            if time_str:
                log_message += f" [{time_str}]"

            # 只在检测耗时较长时输出日志
            if total_elapsed > 0.05:  # 只记录耗时较长的检测
                log_message += f" - 总耗时:{total_elapsed:.3f}s"
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
        start_time = time.time()

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

            elapsed = time.time() - start_time
            self.log(f"音频提取耗时: {elapsed:.2f}秒")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"提取音频失败: {e.stderr}")
            return False

    def split_video_by_keyframes(self, video_path, segment_duration, output_dir):
        """按关键帧分割视频"""
        self.log(f"开始分割视频: {os.path.basename(video_path)}")
        start_time = time.time()

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

            elapsed = time.time() - start_time
            self.log(f"视频分割完成，共{len(segments)}段，耗时: {elapsed:.2f}秒")

        except subprocess.CalledProcessError as e:
            self.log(f"视频分割失败: {e.stderr}")
            return []

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
        model_load_start = time.time()
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

        model_load_end = time.time()
        self.log(f"模型加载总耗时: {model_load_end - model_load_start:.2f}秒")

        return generator

    def process_single_frame(self, frame):
        """处理单帧图像 - 修复内存泄漏版本"""
        start_time = time.time()

        if self.test_mode_var.get():
            # 测试模式不处理，直接返回RGB格式
            elapsed = time.time() - start_time
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 返回RGB格式

        from torchvision.transforms import ToTensor

        # 预处理阶段时间统计
        preprocess_start = time.time()

        # 预处理 - 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        original_h, original_w = h, w

        # 下采样（如果需要）
        scale = int(self.scale_var.get())
        downsample_threshold = int(self.downsample_threshold.get())

        short_side = min(h, w)

        if downsample_threshold != -1 and short_side > downsample_threshold:
            rescale_factor = short_side / downsample_threshold
            new_w = int(w / rescale_factor)
            new_h = int(h / rescale_factor)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # 立即清理中间变量
            del frame
            frame = None

        # 裁剪（如果需要）
        if self.crop_for_4x_var.get() and scale == 4:
            h, w, _ = frame_rgb.shape
            if h % 4 != 0:
                frame_rgb = frame_rgb[:4 * (h // 4), :, :]
            if w % 4 != 0:
                frame_rgb = frame_rgb[:, :4 * (w // 4), :]

        preprocess_time = time.time() - preprocess_start

        # 推理阶段时间统计
        inference_start = time.time()

        # 转换为tensor并进行推理
        img_tensor = ToTensor()(frame_rgb).unsqueeze(0)  # 形状: [1, 3, H, W]

        # 立即清理不再需要的变量
        del frame_rgb
        frame_rgb = None

        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        img_tensor = img_tensor.to(dtype=self.weight_dtype)

        # 推理
        with torch.no_grad():
            result = self.generator(img_tensor)

        inference_time = time.time() - inference_start

        # 后处理阶段时间统计
        postprocess_start = time.time()

        # 将结果移动到CPU，并释放GPU内存
        result_cpu = result[0].cpu().detach()

        # 立即清理GPU变量
        del img_tensor
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 转换为numpy数组，调整通道顺序，并缩放到0-255
        result_np = result_cpu.numpy()
        result_np = np.transpose(result_np, (1, 2, 0))  # 从 [C, H, W] 转换为 [H, W, C]
        result_np = np.clip(result_np * 255.0, 0, 255).astype(np.uint8)

        # 清理中间变量
        del result_cpu

        # 如果需要，缩放回原始大小
        if downsample_threshold != -1 and short_side > downsample_threshold:
            output_h = int(original_h * scale)
            output_w = int(original_w * scale)
            result_np = cv2.resize(result_np, (output_w, output_h), interpolation=cv2.INTER_LINEAR)

        postprocess_time = time.time() - postprocess_start
        total_elapsed = time.time() - start_time

        # 记录详细的时间统计（只记录耗时较长的帧处理）
        if total_elapsed > 0.2:  # 只记录超过200ms的帧处理
            self.log(f"帧处理耗时: {total_elapsed:.3f}s [预处理:{preprocess_time:.3f}s, "
                     f"推理:{inference_time:.3f}s, 后处理:{postprocess_time:.3f}s]")

        return result_np

    def process_frame_with_enhanced_dup_detect(self, frame, frame_idx):
        """处理单帧，包含增强的重复帧检测 - 修复内存泄漏版本"""
        start_time = time.time()
        is_duplicate = False

        try:
            # 检查是否为重复帧
            is_duplicate, matched_sr_result, current_hash, current_thumbnail = \
                self.check_frame_duplicate_enhanced(frame, frame_idx)

            if is_duplicate and matched_sr_result is not None:
                # 找到重复帧，直接使用历史超分辨率结果
                result_np = matched_sr_result.copy()  # 创建副本

                # 计算当前帧的信息（如果需要）
                if current_hash is None and self.use_hash_var.get():
                    current_hash, _ = self.calculate_frame_hash(frame)
                if current_thumbnail is None and self.use_ssim_var.get():
                    h, w = frame.shape[:2]
                    if h > 180 or w > 320:
                        new_h = 180
                        new_w = int(w * (180 / h))
                        current_thumbnail = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        current_thumbnail = frame.copy()

                # 添加帧到历史记录
                self.add_frame_to_history(frame, current_hash, current_thumbnail, result_np, frame_idx)

                total_elapsed = time.time() - start_time
                return result_np, current_hash, current_thumbnail, is_duplicate

            # 非重复帧，进行超分辨率处理
            process_start = time.time()
            result_np = self.process_single_frame(frame)
            process_time = time.time() - process_start

            # 计算当前帧的信息
            if self.use_hash_var.get():
                hash_start = time.time()
                if current_hash is None:
                    current_hash, hash_time = self.calculate_frame_hash(frame)
                else:
                    hash_time = time.time() - hash_start
            else:
                hash_time = 0

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

            total_time = time.time() - start_time

            # 定期清理内存
            if frame_idx % 50 == 0:
                self.cleanup_memory()

            if total_time > 0.3:  # 只记录耗时较长的帧处理
                self.log(f"帧 {frame_idx:04d}: 超分处理耗时: {process_time:.3f}s，总耗时: {total_time:.3f}s")

            return result_np, current_hash, current_thumbnail, is_duplicate

        except Exception as e:
            self.log(f"帧 {frame_idx} 处理出错: {e}")
            # 发生错误时清理内存
            self.cleanup_memory()
            raise

    def detect_progress_from_folders(self):
        """从文件夹内容检测进度"""
        if not self.temp_base_dir or not os.path.exists(self.temp_base_dir):
            return 0, 0, []

        self.log("开始从文件夹检测进度...")

        # 1. 从04_processed_segments文件夹获取已处理的片段
        processed_dir = os.path.join(self.temp_base_dir, "04_processed_segments")
        processed_segments = []
        if os.path.exists(processed_dir):
            for f in os.listdir(processed_dir):
                if f.startswith("processed_segment_") and f.endswith(".mp4"):
                    # 提取片段编号，如processed_segment_001.mp4 -> 1
                    try:
                        segment_num = int(f.split('_')[2].split('.')[0])
                        processed_segments.append(segment_num)
                    except:
                        pass

        # 2. 从03_segment_frames文件夹获取当前处理的片段和帧
        frames_dir = os.path.join(self.temp_base_dir, "03_segment_frames")
        current_segment = 0
        current_frame = 0

        if os.path.exists(frames_dir):
            # 查找所有after文件夹
            after_dirs = []
            for item in os.listdir(frames_dir):
                item_path = os.path.join(frames_dir, item)
                if os.path.isdir(item_path) and item.endswith("_after"):
                    after_dirs.append(item_path)

            if after_dirs:
                # 按文件夹名排序（最新的在前）
                after_dirs.sort(key=lambda x: os.path.basename(x))

                # 处理最新的after文件夹
                latest_after_dir = after_dirs[-1]
                dir_name = os.path.basename(latest_after_dir)

                # 提取片段名称，如segment_000_after -> segment_000
                try:
                    current_segment_name = dir_name.replace("_after", "")
                    # 从segment_000获取数字部分
                    if current_segment_name.startswith("segment_"):
                        try:
                            current_segment = int(current_segment_name.split('_')[1])
                        except:
                            current_segment = 0
                except:
                    current_segment = 0

                # 计算已处理的帧数
                if os.path.exists(latest_after_dir):
                    frame_files = [f for f in os.listdir(latest_after_dir)
                                   if f.startswith("frame_") and f.endswith(".png")]
                    if frame_files:
                        # 按文件名排序，获取最大的帧号
                        frame_files.sort()
                        last_frame = frame_files[-1]
                        try:
                            current_frame = int(last_frame.split('_')[1].split('.')[0]) + 1
                        except:
                            current_frame = 0

        # 3. 确定下一个要处理的片段
        if processed_segments:
            last_processed = max(processed_segments)
            next_segment = last_processed + 1
        else:
            next_segment = 1

        # 如果当前片段已经有帧在处理，使用当前片段
        if current_frame > 0:
            next_segment = current_segment

        self.log(f"进度检测结果: 下一个片段={next_segment}, 当前帧={current_frame}")
        return next_segment, current_frame, processed_segments

    def process_segment_frames(self, segment_path, segment_index):
        """处理视频片段的所有帧（逐帧处理）- 修复：始终生成视频片段并删除临时帧文件"""
        segment_name = os.path.basename(segment_path)
        self.log(f"处理片段 {segment_index}: {segment_name}")
        segment_start_time = time.time()

        if self.test_mode_var.get():
            self.log("测试模式：仅进行重复帧检测，不进行超分辨率处理")
            # 测试模式不生成视频，但会保留帧文件供检查
            return None, None

        # 初始化历史帧缓存
        self.init_history_cache()

        # 更新重复帧计数（已重置为0）
        self.update_dup_info(self.dup_frame_count)

        # 为当前片段创建帧目录（直接创建在03_segment_frames下）
        setup_start = time.time()
        before_dir, after_dir = self.setup_segment_frame_dirs(segment_path)
        setup_time = time.time() - setup_start

        if not before_dir or not after_dir:
            self.log("错误：无法创建帧目录")
            return None, None

        self.log(f"目录设置耗时: {setup_time:.2f}秒")

        # 提取音频
        audio_name = segment_name.replace('.mp4', '.aac')
        audio_path = os.path.join(self.temp_base_dir, "02_audio", audio_name)

        audio_start = time.time()
        has_audio = self.extract_audio(segment_path, audio_path)
        audio_time = time.time() - audio_start

        if has_audio:
            self.log(f"音频提取成功，耗时: {audio_time:.2f}秒")
        else:
            self.log("视频无音频或音频提取失败")

        # 读取视频
        cap_start = time.time()
        cap = cv2.VideoCapture(segment_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_time = time.time() - cap_start

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

        self.log(f"视频信息获取耗时: {cap_time:.2f}秒")
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
                self.log(f"重复帧检测: {', '.join(methods)}，历史帧: {history_size}")
            else:
                self.log(f"重复帧检测: {', '.join(methods)}，仅与前帧比较")

        # 从进度检测中获取起始帧
        start_frame = self.current_frame_in_segment
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.log(f"从第 {start_frame + 1} 帧恢复处理")

        frame_idx = start_frame
        frame_files = []

        # 注意：已删除重复帧记录文件的创建

        # 初始化帧计数器
        frames_processed = 0
        segment_dup_count = 0  # 当前片段的重复帧数

        # 统计计时
        total_frame_time = 0
        total_dup_detect_time = 0
        total_sr_time = 0
        total_io_time = 0

        while True:
            # 检查是否被停止
            if self.stopped:
                self.log(f"停止处理：片段 {segment_index} 的第 {frame_idx + 1} 帧")
                self.log(f"已处理的帧已保存在: {after_dir}")
                break

            # 检查是否暂停 - 使用高效等待
            if self.paused:
                self.log(f"处理暂停于片段 {segment_index} 的第 {frame_idx + 1} 帧")

                # 释放GPU内存以降低占用
                if self.generator is not None:
                    try:
                        # 将模型移动到CPU并释放GPU内存
                        self.generator = self.generator.cpu()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # 等待CUDA操作完成
                        self.log("模型已移动到CPU，GPU内存已释放")
                    except Exception as e:
                        self.log(f"移动模型到CPU时出错: {e}")

                # 高效等待，而不是忙等待
                self.pause_btn.config(text="▶ 继续")
                self.update_status("已暂停", "orange")

                while self.paused and not self.stopped:
                    time.sleep(0.5)  # 使用较短的休眠时间以便快速响应

                # 恢复处理
                if not self.stopped:
                    if self.generator is not None:
                        try:
                            # 将模型移回GPU
                            self.generator = self.generator.cuda()
                            self.log("模型已移回GPU")
                        except Exception as e:
                            self.log(f"移动模型回GPU时出错: {e}")

                    self.pause_btn.config(text="⏸ 暂停")
                    self.update_status("处理中...", "blue")
                    self.log(f"处理继续于片段 {segment_index} 的第 {frame_idx + 1} 帧")

                if self.stopped:
                    break

            # 每处理50帧清理一次内存
            if frame_idx % 50 == 0:
                self.cleanup_memory()
                self.log(f"已清理内存（处理到第 {frame_idx} 帧）")

            # 每处理100帧清理一次历史缓存
            if frame_idx % 100 == 0 and self.enable_dup_detect_var.get():
                self.clear_history_cache()
                self.init_history_cache()  # 重新初始化
                self.log(f"已清空历史缓存（处理到第 {frame_idx} 帧）")

            # 读取帧
            read_start = time.time()
            ret, frame = cap.read()
            read_time = time.time() - read_start
            total_io_time += read_time

            if not ret:
                break

            # 保存原始帧到before目录
            save_start = time.time()
            before_path = os.path.join(before_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(before_path, frame)
            save_time = time.time() - save_start
            total_io_time += save_time

            # 使用增强的重复帧检测处理帧
            process_start = time.time()
            sr_frame, current_hash, current_thumbnail, is_duplicate = \
                self.process_frame_with_enhanced_dup_detect(frame, frame_idx)
            frame_process_time = time.time() - process_start

            if is_duplicate:
                total_dup_detect_time += frame_process_time
                segment_dup_count += 1
            else:
                total_sr_time += frame_process_time

            total_frame_time += frame_process_time

            # 保存处理后的帧到after目录
            save_sr_start = time.time()
            after_path = os.path.join(after_dir, f"frame_{frame_idx:06d}.png")
            sr_frame_bgr = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(after_path, sr_frame_bgr)
            save_sr_time = time.time() - save_sr_start
            total_io_time += save_sr_time

            # 添加到帧文件列表
            frame_files.append(after_path)

            # 注意：已删除重复帧记录文件的写入

            # 更新当前帧
            self.current_frame_in_segment = frame_idx + 1
            frames_processed += 1

            # 更新详细进度
            self.update_detailed_progress(self.current_frame_in_segment, total_frames)

            # 每处理10帧更新一次进度
            if frames_processed % 10 == 0:
                progress = (self.current_frame_in_segment / total_frames) * 100
                self.update_progress(progress)

            frame_idx += 1

        cap.release()

        # 记录片段处理统计
        segment_elapsed = time.time() - segment_start_time
        avg_frame_time = total_frame_time / max(frames_processed, 1) if frames_processed > 0 else 0

        self.log("=" * 60)
        self.log(f"片段处理完成统计:")
        self.log(f"  片段名称: {segment_name}")
        self.log(f"  总耗时: {segment_elapsed:.2f}秒")
        self.log(f"  处理帧数: {frames_processed}")
        if frames_processed > 0:
            self.log(f"  平均每帧耗时: {avg_frame_time:.3f}秒")
            self.log(
                f"  重复帧检测耗时: {total_dup_detect_time:.2f}秒 ({total_dup_detect_time / segment_elapsed * 100:.1f}%)")
            self.log(f"  超分辨率处理耗时: {total_sr_time:.2f}秒 ({total_sr_time / segment_elapsed * 100:.1f}%)")
            self.log(f"  文件IO耗时: {total_io_time:.2f}秒 ({total_io_time / segment_elapsed * 100:.1f}%)")

        if self.enable_dup_detect_var.get():
            dup_percentage = (segment_dup_count / frames_processed * 100) if frames_processed > 0 else 0
            self.log(f"  检测到重复帧: {segment_dup_count}个 ({dup_percentage:.1f}%)")
            if segment_dup_count > 0:
                self.log(f"  重复帧节省时间估算: {segment_dup_count * avg_frame_time:.2f}秒")

        # 清空历史缓存以释放内存
        self.clear_history_cache()

        # 只有在片段完全处理完且没有停止时才生成视频
        if not self.stopped and frame_idx >= total_frames and frame_files:
            # 生成处理后的片段视频
            processed_segment_path = os.path.join(self.temp_base_dir, "04_processed_segments",
                                                  f"processed_{segment_name}")

            # 将帧转换为视频
            encode_start = time.time()
            success = self.frames_to_video(frame_files, processed_segment_path, fps, output_width, output_height,
                                           audio_path)
            encode_time = time.time() - encode_start

            if success:
                self.log(f"片段视频编码耗时: {encode_time:.2f}秒")
                self.log(f"片段视频生成成功: {processed_segment_path}")

                # 清理当前片段的帧目录
                self.cleanup_segment_frame_dirs(segment_path)
                self.log(f"已清理片段 {segment_name} 的帧临时文件 (before和after目录)")

                # 如果有立即合并功能，调用合并
                if self.immediate_merge_var.get() and not self.test_mode_var.get():
                    self.update_immediate_merge()

                return processed_segment_path, audio_path
            else:
                self.log("片段视频生成失败")
                return None, None
        else:
            if self.stopped:
                self.log("处理被停止，不生成视频片段，保留临时文件以便下次继续处理")
            elif not frame_files:
                self.log("没有帧文件可处理")
            return None, None

    def check_opencv_encoder_support(self):
        """检查OpenCV编码器支持"""
        test_size = (100, 100)
        test_encoders = ['mp4v', 'avc1', 'MJPG', 'XVID']

        for encoder in test_encoders:
            try:
                fourcc = cv2.VideoWriter_fourcc(*encoder)
                out = cv2.VideoWriter(tempfile.mktemp(suffix='.mp4'), fourcc, 1, test_size)
                if out.isOpened():
                    out.release()
                    return True
            except:
                pass
        return False

    def frames_to_video(self, frame_files, output_path, fps, width, height, audio_path=None):
        """将帧序列转换为视频"""
        encoder_mode = self.video_encoder_mode.get()

        if encoder_mode == "ffmpeg" or (encoder_mode == "auto" and not self.check_opencv_encoder_support()):
            return self.frames_to_video_alternative(frame_files, output_path, fps, width, height, audio_path)
        else:
            return self.frames_to_video_opencv(frame_files, output_path, fps, width, height, audio_path)

    def frames_to_video_opencv(self, frame_files, output_path, fps, width, height, audio_path=None):
        """将帧序列转换为视频（使用OpenCV）"""
        self.log(f"正在生成视频: {output_path}")
        start_time = time.time()

        if not frame_files:
            self.log("错误: 没有可用的帧文件")
            return False

        # 创建临时视频文件（无音频）
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')

        # 尝试多种编码器，避免OpenH264问题
        encoders_to_try = [
            ('mp4v', 'mp4'),  # MPEG-4 编码
            ('MJPG', 'avi'),  # Motion JPEG
            ('XVID', 'avi'),  # XVID 编码
            ('I420', 'avi'),  # YUV 编码
            ('IYUV', 'avi'),  # YUV 编码
            ('DIVX', 'avi')  # DivX 编码
        ]

        out = None
        selected_encoder = None
        selected_ext = None

        # 尝试不同的编码器
        for codec, ext in encoders_to_try:
            try:
                if ext == 'avi':
                    temp_video_path = output_path.replace('.mp4', '_temp.avi')
                else:
                    temp_video_path = output_path.replace('.mp4', '_temp.mp4')

                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

                if out.isOpened():
                    selected_encoder = codec
                    selected_ext = ext
                    self.log(f"使用编码器: {codec}，文件格式: {ext}")
                    break
                else:
                    out.release()
            except Exception as e:
                if out:
                    out.release()
                continue

        if not out or not out.isOpened():
            self.log("错误: 无法创建视频写入器，尝试使用ffmpeg方式")
            return self.frames_to_video_alternative(frame_files, output_path, fps, width, height, audio_path)

        # 按顺序写入所有帧
        write_start = time.time()
        frame_count = 0
        read_time = 0
        write_time = 0

        for frame_file in sorted(frame_files):
            if os.path.exists(frame_file):
                read_start = time.time()
                frame = cv2.imread(frame_file)
                read_time += time.time() - read_start

                if frame is not None:
                    # 确保帧的大小与视频写入器匹配
                    if frame.shape[1] != width or frame.shape[0] != height:
                        resize_start = time.time()
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                        read_time += time.time() - resize_start

                    # 确保帧是8位无符号整数
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)

                    write_frame_start = time.time()
                    out.write(frame)
                    write_time += time.time() - write_frame_start

                    frame_count += 1

                    # 每100帧输出一次进度
                    if frame_count % 100 == 0:
                        current_time = time.time() - write_start
                        avg_time_per_frame = current_time / frame_count
                        self.log(f"已写入 {frame_count} 帧，平均每帧: {avg_time_per_frame:.3f}秒")

        write_total_time = time.time() - write_start
        out.release()

        # 确保视频文件创建成功
        if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
            self.log(f"错误: 视频文件创建失败: {temp_video_path}")
            return self.frames_to_video_alternative(frame_files, output_path, fps, width, height, audio_path)

        file_size = os.path.getsize(temp_video_path) / (1024 * 1024)  # 转换为MB
        self.log(f"临时视频创建成功，大小: {file_size:.2f} MB，写入耗时: {write_total_time:.2f}秒")
        self.log(f"  读取耗时: {read_time:.2f}秒，写入耗时: {write_time:.2f}秒")

        # 如果有音频，合并音频和视频
        if audio_path and os.path.exists(audio_path):
            self.log("合并音频和视频...")
            merge_start = time.time()

            try:
                # 如果生成的是AVI文件，需要转换格式
                if selected_ext == 'avi':
                    # 先转换为mp4
                    mp4_temp = temp_video_path.replace('.avi', '_converted.mp4')
                    convert_cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_video_path,
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '23',
                        '-loglevel', 'quiet',
                        mp4_temp
                    ]

                    subprocess.run(convert_cmd, check=True, capture_output=True, text=True)

                    if os.path.exists(mp4_temp):
                        # 删除原始AVI文件
                        os.remove(temp_video_path)
                        temp_video_path = mp4_temp

                # 合并音频
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-strict', 'experimental',
                    '-loglevel', 'quiet',
                    output_path
                ]

                result = subprocess.run(cmd, check=True, capture_output=True, text=True)

                # 删除临时文件
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

                merge_time = time.time() - merge_start
                total_time = time.time() - start_time

                self.log(f"音频视频合并成功，耗时: {total_time:.2f}秒")
                self.log(f"  详细时间: 写入帧{write_total_time:.2f}s, 合并{merge_time:.2f}s")
                self.log(f"生成视频: {output_path}，分辨率: {width}x{height}，帧率: {fps}，帧数: {frame_count}")
                return True

            except subprocess.CalledProcessError as e:
                self.log(f"音频视频合并失败: {e.stderr}")

                # 如果合并失败，尝试另一种方法
                try:
                    self.log("尝试第二种方法合并音频视频...")
                    cmd2 = [
                        'ffmpeg', '-y',
                        '-i', temp_video_path,
                        '-i', audio_path,
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '23',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-strict', 'experimental',
                        '-loglevel', 'quiet',
                        output_path
                    ]

                    subprocess.run(cmd2, check=True, capture_output=True, text=True)

                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)

                    self.log("第二种方法合并成功")
                    return True
                except Exception as e2:
                    self.log(f"第二种方法也失败: {e2}")
                    # 如果合并失败，使用临时视频文件作为输出
                    if os.path.exists(temp_video_path):
                        shutil.move(temp_video_path, output_path)
                        self.log("使用无音频视频作为输出")
                    return True

        else:
            # 如果没有音频，直接使用临时视频文件
            if os.path.exists(temp_video_path):
                # 如果是AVI格式，转换为MP4
                if selected_ext == 'avi':
                    self.log("将AVI转换为MP4...")
                    convert_cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_video_path,
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '23',
                        '-loglevel', 'quiet',
                        output_path
                    ]

                    try:
                        subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
                        os.remove(temp_video_path)
                    except Exception as e:
                        self.log(f"转换失败: {e}")
                        shutil.move(temp_video_path, output_path)
                else:
                    shutil.move(temp_video_path, output_path)

                total_time = time.time() - start_time
                self.log(f"视频生成成功，总耗时: {total_time:.2f}秒")
                self.log(f"生成视频: {output_path}，分辨率: {width}x{height}，帧率: {fps}，帧数: {frame_count}")
                return True
            else:
                self.log(f"错误: 视频文件未创建: {temp_video_path}")

        return False

    def frames_to_video_alternative(self, frame_files, output_path, fps, width, height, audio_path=None):
        """替代方法：使用ffmpeg直接生成视频（避免OpenCV编码器问题）"""
        self.log("使用ffmpeg直接生成视频...")
        start_time = time.time()

        if not frame_files:
            self.log("错误: 没有可用的帧文件")
            return False

        # 创建临时文件列表
        list_file = tempfile.mktemp(suffix=".txt")
        frame_files_sorted = sorted(frame_files)

        with open(list_file, 'w', encoding='utf-8') as f:
            for frame_file in frame_files_sorted:
                f.write(f"file '{os.path.abspath(frame_file)}'\n")

        # 使用ffmpeg从图像序列生成视频
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-r', str(fps),
            '-i', list_file,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-loglevel', 'quiet',
            temp_video_path
        ]

        try:
            convert_start = time.time()
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            convert_time = time.time() - convert_start

            if not os.path.exists(temp_video_path):
                self.log("错误: ffmpeg未能生成视频")
                os.remove(list_file)
                return False

            # 如果有音频，合并音频
            if audio_path and os.path.exists(audio_path):
                merge_cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-loglevel', 'quiet',
                    output_path
                ]

                merge_start = time.time()
                subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
                merge_time = time.time() - merge_start

                os.remove(temp_video_path)
                total_time = time.time() - start_time
                self.log(
                    f"视频生成成功，总耗时: {total_time:.2f}秒 (转换: {convert_time:.2f}s, 合并: {merge_time:.2f}s)")
            else:
                shutil.move(temp_video_path, output_path)
                total_time = time.time() - start_time
                self.log(f"视频生成成功，总耗时: {total_time:.2f}秒 (转换: {convert_time:.2f}s)")

            os.remove(list_file)
            return True

        except subprocess.CalledProcessError as e:
            self.log(f"ffmpeg生成视频失败: {e.stderr}")
            if os.path.exists(list_file):
                os.remove(list_file)
            return False

    def process_segment_directly(self, segment_path, segment_index):
        """直接处理视频片段（不进行重复帧检测）"""
        segment_name = os.path.basename(segment_path)
        self.log(f"直接处理片段 {segment_index}: {segment_name}")
        segment_start_time = time.time()

        if self.test_mode_var.get():
            self.log("测试模式：不进行超分辨率处理")
            return None, None

        # 读取视频
        try:
            video = VideoFileClip(segment_path)
        except Exception as e:
            self.log(f"读取视频失败: {e}")
            return None, None

        # 获取视频信息
        fps = video.fps
        width, height = video.size
        total_frames = int(video.duration * fps)
        has_audio = video.audio is not None

        # 提取音频（如果有）
        audio_path = None
        if has_audio:
            audio_name = segment_name.replace('.mp4', '.aac')
            audio_path = os.path.join(self.temp_base_dir, "02_audio", audio_name)
            try:
                video.audio.write_audiofile(audio_path, verbose=False)
                self.log("音频提取成功")
            except Exception as e:
                self.log(f"音频提取失败: {e}")
                audio_path = None
                has_audio = False

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
        self.log(f"直接处理模式：使用moviepy处理，不进行重复帧检测")

        # 创建输出路径
        processed_segment_path = os.path.join(self.temp_base_dir, "04_processed_segments", f"processed_{segment_name}")

        # 创建视频写入器
        try:
            if has_audio and audio_path:
                writer = FFMPEG_VideoWriter(processed_segment_path, (output_width, output_height), fps,
                                            audiofile=audio_path)
                self.log("使用带音频的视频写入器")
            else:
                writer = FFMPEG_VideoWriter(processed_segment_path, (output_width, output_height), fps)
                self.log("使用无音频的视频写入器")
        except Exception as e:
            self.log(f"创建视频写入器失败: {e}")
            video.close()
            return None, None

        frame_idx = 0
        frames_processed = 0
        total_frame_time = 0

        # 处理每一帧
        for frame_idx, img_lr in enumerate(video.iter_frames(fps=fps, dtype='uint8')):
            # 检查是否被停止
            if self.stopped:
                self.log(f"停止处理：片段 {segment_index} 的第 {frame_idx + 1} 帧")
                break

            # 直接处理模式不支持暂停，所以不需要检查暂停状态

            # 注意：moviepy返回的是RGB格式，需要转换为BGR进行超分处理
            # 转换为BGR格式
            img_lr_bgr = cv2.cvtColor(img_lr, cv2.COLOR_RGB2BGR)

            # 下采样（如果需要）
            if rescale_factor != 1:
                img_lr_bgr = cv2.resize(img_lr_bgr, (int(width / rescale_factor), int(height / rescale_factor)),
                                        interpolation=cv2.INTER_LINEAR)

            # 裁剪（如果需要）
            if self.crop_for_4x_var.get() and scale == 4:
                h, w, _ = img_lr_bgr.shape
                if h % 4 != 0:
                    img_lr_bgr = img_lr_bgr[:4 * (h // 4), :, :]
                if w % 4 != 0:
                    img_lr_bgr = img_lr_bgr[:, :4 * (w // 4), :]

            # 处理帧
            process_start = time.time()
            sr_frame = self.process_single_frame(img_lr_bgr)
            frame_process_time = time.time() - process_start
            total_frame_time += frame_process_time

            # 写入帧（注意：moviepy需要RGB格式）
            sr_frame_rgb = cv2.cvtColor(sr_frame, cv2.COLOR_BGR2RGB)
            writer.write_frame(sr_frame_rgb)

            # 更新进度
            frames_processed += 1

            # 更新详细进度
            if total_frames > 0:
                self.update_detailed_progress(frame_idx + 1, total_frames)

            # 更新进度条
            if total_frames > 0:
                progress = ((frame_idx + 1) / total_frames) * 100
                self.update_progress(progress)

            # 每处理10帧更新一次日志
            if (frame_idx + 1) % 10 == 0:
                avg_frame_time = total_frame_time / (frame_idx + 1)
                self.log(f"已处理 {frame_idx + 1}/{total_frames} 帧，平均每帧耗时: {avg_frame_time:.3f}秒")

            # 每处理50帧清理一次内存
            if frame_idx % 50 == 0:
                self.cleanup_memory()

        # 关闭写入器和视频
        writer.close()
        video.close()

        # 记录片段处理统计
        segment_elapsed = time.time() - segment_start_time
        avg_frame_time = total_frame_time / max(frames_processed, 1)

        self.log(f"直接处理完成: {segment_name}，总耗时: {segment_elapsed:.2f}秒")
        self.log(f"  处理帧数: {frames_processed}，平均每帧耗时: {avg_frame_time:.3f}秒")

        # 如果有立即合并功能，调用合并
        if self.immediate_merge_var.get() and not self.test_mode_var.get():
            self.update_immediate_merge()

        return processed_segment_path, audio_path if has_audio else None

    def update_immediate_merge(self):
        """更新立即合并视频 - 检查04_processed_segments文件夹并合并到05_immediate_merge"""
        if not self.immediate_merge_var.get() or self.test_mode_var.get():
            return None

        start_time = time.time()

        # 获取路径
        processed_segments_dir = os.path.join(self.temp_base_dir, "04_processed_segments")
        merge_dir = os.path.join(self.temp_base_dir, "05_immediate_merge")
        log_file_path = os.path.join(processed_segments_dir, "merge_log.txt")

        # 确保目录存在
        os.makedirs(merge_dir, exist_ok=True)

        # 读取日志文件，获取已合并的片段
        merged_segments = set()
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        segment_name = line.strip()
                        if segment_name:
                            merged_segments.add(segment_name)
                self.log(f"从日志文件中读取到 {len(merged_segments)} 个已合并的片段")
            except Exception as e:
                self.log(f"读取合并日志失败: {e}")
                merged_segments = set()

        # 获取04_processed_segments目录下所有的processed_segment_*.mp4文件
        all_processed_segments = []
        if os.path.exists(processed_segments_dir):
            for f in sorted(os.listdir(processed_segments_dir)):
                if f.startswith("processed_segment_") and f.endswith(".mp4"):
                    all_processed_segments.append(f)

        if not all_processed_segments:
            self.log("没有找到已处理的片段")
            return None

        # 找出未合并的片段
        unmerged_segments = []
        for segment in all_processed_segments:
            if segment not in merged_segments:
                unmerged_segments.append(segment)

        if not unmerged_segments:
            self.log("所有片段都已合并，无需操作")
            return None

        self.log(f"找到 {len(unmerged_segments)} 个未合并的片段")

        # 获取当前的合并视频（如果有）
        merged_videos = []
        if os.path.exists(merge_dir):
            for f in os.listdir(merge_dir):
                if f.startswith("merged_video") and f.endswith(".mp4"):
                    merged_videos.append(os.path.join(merge_dir, f))

        merged_video_path = None
        if merged_videos:
            # 按修改时间排序，取最新的
            merged_videos.sort(key=lambda x: os.path.getmtime(x))
            merged_video_path = merged_videos[-1]
            self.log(f"找到现有的合并视频: {os.path.basename(merged_video_path)}")
        else:
            self.log("没有找到现有的合并视频，将创建新的")

        # 构建要合并的视频文件列表
        video_files_to_merge = []

        # 如果有现有的合并视频，先加入
        if merged_video_path and os.path.exists(merged_video_path):
            video_files_to_merge.append(merged_video_path)

        # 加入新的未合并片段
        for segment in unmerged_segments:
            segment_path = os.path.join(processed_segments_dir, segment)
            if os.path.exists(segment_path):
                video_files_to_merge.append(segment_path)
            else:
                self.log(f"警告：片段文件不存在: {segment}")

        if len(video_files_to_merge) == 0:
            self.log("没有视频文件可合并")
            return None

        # 如果只有一个文件且是新的合并视频，直接复制
        if len(video_files_to_merge) == 1 and video_files_to_merge[0] == merged_video_path:
            self.log("只有现有的合并视频，无需操作")
            return merged_video_path

        # 创建临时文件列表
        list_file = tempfile.mktemp(suffix=".txt")

        try:
            with open(list_file, 'w', encoding='utf-8') as f:
                for video_file in video_files_to_merge:
                    f.write(f"file '{os.path.abspath(video_file)}'\n")

            # 生成新的合并视频文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_merged_video_path = os.path.join(merge_dir, f"merged_video_{timestamp}.mp4")

            # 使用ffmpeg合并
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c', 'copy',
                new_merged_video_path
            ]

            merge_start = time.time()
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            merge_time = time.time() - merge_start

            # 更新日志文件，记录新合并的片段
            try:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    for segment in unmerged_segments:
                        f.write(f"{segment}\n")
                self.log(f"已将 {len(unmerged_segments)} 个片段记录到合并日志")
            except Exception as e:
                self.log(f"写入合并日志失败: {e}")

            # 删除旧的合并视频（如果有）
            if merged_video_path and merged_video_path != new_merged_video_path:
                try:
                    os.remove(merged_video_path)
                    self.log(f"已删除旧的合并视频: {os.path.basename(merged_video_path)}")
                except Exception as e:
                    self.log(f"删除旧的合并视频失败: {e}")

            elapsed = time.time() - start_time
            self.log(f"立即合成成功: 合并了 {len(unmerged_segments)} 个新片段，耗时: {elapsed:.2f}秒 (合并: {merge_time:.2f}s)")
            self.log(f"新的合并视频: {os.path.basename(new_merged_video_path)}")

            return new_merged_video_path

        except Exception as e:
            self.log(f"立即合成失败: {e}")
            return None
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)

    def process_single_video(self, video_path):
        """处理单个视频"""
        try:
            self.log(
                f"开始处理视频 {self.current_video_index + 1}/{len(self.input_paths)}: {os.path.basename(video_path)}")
            video_start_time = time.time()

            # 设置视频基础名称
            self.video_base_name = Path(video_path).stem

            # 设置临时目录（基于视频文件名和测试模式）
            temp_dirs = self.setup_temp_dirs(video_path)
            if not temp_dirs:
                raise ValueError("无法创建临时目录")

            self.log(f"临时文件目录: {temp_dirs['base']}")
            if self.is_test_mode_folder:
                self.log("注意：当前为测试模式文件夹")

            # 检测进度
            if self.enable_dup_detect_var.get():
                # 启用重复帧检测模式：从03和04文件夹检测进度
                next_segment, current_frame, processed_segments = self.detect_progress_from_folders()
                self.current_segment_index = next_segment - 1 if next_segment > 0 else 0
                self.current_frame_in_segment = current_frame
                self.processed_segments = [f"segment_{i:03d}.mp4" for i in processed_segments]
                self.log(f"从文件夹检测到进度: 下一个片段={next_segment}, 当前帧={current_frame}")
            else:
                # 直接处理模式：只从04文件夹检测进度
                processed_dir = os.path.join(self.temp_base_dir, "04_processed_segments")
                if os.path.exists(processed_dir):
                    processed_files = [f for f in os.listdir(processed_dir)
                                       if f.startswith("processed_") and f.endswith(".mp4")]
                    if processed_files:
                        # 提取最后一个处理文件的片段编号
                        last_processed = sorted(processed_files)[-1]
                        try:
                            # 格式如：processed_segment_001.mp4
                            segment_num = int(last_processed.split('_')[2].split('.')[0])
                            self.current_segment_index = segment_num  # 下一个要处理的片段
                            self.processed_segments = [f"segment_{i:03d}.mp4" for i in range(1, segment_num)]
                            self.log(f"从04文件夹检测到进度: 已处理{segment_num}个片段，下一个片段={segment_num + 1}")
                        except:
                            self.current_segment_index = 0
                            self.processed_segments = []
                            self.log("无法解析处理文件名，从头开始处理")
                    else:
                        self.current_segment_index = 0
                        self.processed_segments = []
                        self.log("未找到已处理的片段，从头开始处理")
                else:
                    self.current_segment_index = 0
                    self.processed_segments = []
                    self.log("未找到04文件夹，从头开始处理")

            # 重置重复帧计数（每个视频开始时重置）
            self.dup_frame_count = 0
            self.update_dup_info(self.dup_frame_count)

            # 步骤1: 加载模型
            if not self.test_mode_var.get():
                self.log("=" * 60)
                self.log("步骤1: 加载模型...")
                self.update_progress(0)
                model_load_start = time.time()
                self.generator = self.load_model()
                model_load_time = time.time() - model_load_start
                self.log(f"模型加载完成，耗时: {model_load_time:.2f}秒")
                self.update_progress(5)
            else:
                self.log("测试模式：跳过模型加载")
                self.update_progress(5)

            # 步骤2: 分割视频（如果需要）
            segments_dir = os.path.join(self.temp_base_dir, "01_original_segments")
            if os.path.exists(segments_dir):
                # 读取已有的片段
                segment_files = []
                for f in sorted(os.listdir(segments_dir)):
                    if f.startswith("segment_") and f.endswith(".mp4"):
                        segment_files.append(os.path.join(segments_dir, f))

                if segment_files:
                    self.segments = segment_files
                    self.total_segments = len(self.segments)
                    self.log(f"找到 {len(self.segments)} 个已有片段")
                    self.update_progress(10)
                else:
                    # 没有片段，需要分割
                    self.log("=" * 60)
                    self.log("步骤2: 分割视频...")
                    segment_duration = float(self.segment_duration.get())
                    split_start = time.time()
                    self.segments = self.split_video_by_keyframes(video_path, segment_duration, temp_dirs['base'])
                    split_time = time.time() - split_start

                    self.total_segments = len(self.segments)
                    self.log(f"视频分割完成，共{len(self.segments)}段，耗时: {split_time:.2f}秒")
                    self.update_progress(10)

                    if not self.segments:
                        raise ValueError("视频分割失败")

                    # 重置进度
                    self.current_segment_index = 0
                    self.current_frame_in_segment = 0
                    self.processed_segments = []
            else:
                # 需要分割视频
                self.log("=" * 60)
                self.log("步骤2: 分割视频...")
                segment_duration = float(self.segment_duration.get())
                split_start = time.time()
                self.segments = self.split_video_by_keyframes(video_path, segment_duration, temp_dirs['base'])
                split_time = time.time() - split_start

                self.total_segments = len(self.segments)
                self.log(f"视频分割完成，共{len(self.segments)}段，耗时: {split_time:.2f}秒")
                self.update_progress(10)

                if not self.segments:
                    raise ValueError("视频分割失败")

                # 重置进度
                self.current_segment_index = 0
                self.current_frame_in_segment = 0
                self.processed_segments = []

            # 步骤3: 处理视频片段
            self.log("=" * 60)
            self.log("步骤3: 处理视频片段...")

            all_processed_segments = []
            total_segment_time = 0
            total_frames_processed = 0
            total_dup_count = 0

            for i in range(self.current_segment_index, len(self.segments)):
                # 检查是否被停止
                if self.stopped:
                    self.log(f"处理被用户停止于片段 {i + 1}")
                    break

                segment = self.segments[i]
                segment_name = os.path.basename(segment)

                # 检查是否已经处理过（通过04文件夹判断）
                processed_segment_path = os.path.join(temp_dirs['processed_segments'], f"processed_{segment_name}")
                if os.path.exists(processed_segment_path):
                    self.log(f"跳过已处理的片段 {i + 1}/{len(self.segments)}: {segment_name}")
                    self.current_segment_index = i + 1
                    self.current_frame_in_segment = 0
                    all_processed_segments.append(processed_segment_path)
                    continue

                self.log(f"处理片段 {i + 1}/{len(self.segments)}: {segment_name}")
                segment_start = time.time()

                if not self.enable_dup_detect_var.get() and not self.test_mode_var.get():
                    # 直接处理模式（不进行重复帧检测）
                    processed_segment_path, audio_path = self.process_segment_directly(segment, i + 1)
                else:
                    # 逐帧处理模式（带重复帧检测或测试模式）
                    processed_segment_path, audio_path = self.process_segment_frames(segment, i + 1)

                segment_time = time.time() - segment_start
                total_segment_time += segment_time

                # 检查是否被停止
                if self.stopped:
                    break

                if processed_segment_path:
                    # 所有非测试模式都会生成视频片段
                    all_processed_segments.append(processed_segment_path)
                elif self.test_mode_var.get():
                    self.log(f"测试模式：片段 {i + 1} 处理完成，帧文件已保存")

                # 更新进度
                self.current_segment_index = i + 1
                self.current_frame_in_segment = 0

                # 更新总体进度
                overall_progress = 10 + (i + 1) / len(self.segments) * 60
                self.update_progress(overall_progress)

                # 处理完一个片段后清理内存
                self.cleanup_memory()

            if self.stopped:
                self.log(f"处理已停止")
                return False

            # 步骤4: 如果处理了多个片段且不是测试模式，拼接视频
            if not self.test_mode_var.get():
                # 检查是否有立即合成的最终视频
                merge_dir = os.path.join(self.temp_base_dir, "05_immediate_merge")
                if self.immediate_merge_var.get() and os.path.exists(merge_dir):
                    # 查找最新的合并视频
                    merged_files = []
                    for f in os.listdir(merge_dir):
                        if f.startswith("merged_video_") and f.endswith(".mp4"):
                            merged_files.append(os.path.join(merge_dir, f))

                    if merged_files:
                        # 按文件名排序（包含时间戳）
                        merged_files.sort(key=lambda x: os.path.basename(x))
                        latest_merged = merged_files[-1]

                        # 将最终合并视频移动到输出目录
                        output_filename = f"{self.video_base_name}_super_resolved.mp4"
                        final_output = os.path.join(self.output_dir.get(), output_filename)

                        merge_start = time.time()
                        shutil.copy2(latest_merged, final_output)
                        merge_time = time.time() - merge_start

                        self.update_progress(95)
                        self.log(f"使用立即合成的视频作为最终输出: {final_output}，复制耗时: {merge_time:.2f}秒")

                        # 清理立即合成目录
                        shutil.rmtree(merge_dir)
                        self.log("已清理立即合成目录")
                    else:
                        # 如果没有立即合成视频，则使用传统拼接方式
                        self.log("=" * 60)
                        self.log("步骤4: 拼接处理后的视频片段...")

                        if all_processed_segments:
                            output_filename = f"{self.video_base_name}_super_resolved.mp4"
                            final_output = os.path.join(self.output_dir.get(), output_filename)

                            if len(all_processed_segments) > 1:
                                self.concatenate_videos(all_processed_segments, final_output)
                            else:
                                # 如果只有一个片段，直接复制
                                copy_start = time.time()
                                shutil.copy2(all_processed_segments[0], final_output)
                                copy_time = time.time() - copy_start
                                self.log(f"复制单个片段，耗时: {copy_time:.2f}秒")

                            self.update_progress(95)
                            self.log(f"最终输出文件: {final_output}")
                        else:
                            self.log("没有可拼接的片段")
                else:
                    # 传统拼接方式
                    self.log("=" * 60)
                    self.log("步骤4: 拼接处理后的视频片段...")

                    if all_processed_segments:
                        output_filename = f"{self.video_base_name}_super_resolved.mp4"
                        final_output = os.path.join(self.output_dir.get(), output_filename)

                        if len(all_processed_segments) > 1:
                            self.concatenate_videos(all_processed_segments, final_output)
                        else:
                            # 如果只有一个片段，直接复制
                            copy_start = time.time()
                            shutil.copy2(all_processed_segments[0], final_output)
                            copy_time = time.time() - copy_start
                            self.log(f"复制单个片段，耗时: {copy_time:.2f}秒")

                        self.update_progress(95)
                        self.log(f"最终输出文件: {final_output}")
                    else:
                        self.log("没有可拼接的片段")
            else:
                # 测试模式：不生成视频
                self.log("测试模式：跳过视频合成步骤")
                self.update_progress(95)

            # 步骤5: 自动清理临时文件
            self.log("=" * 60)
            self.log("步骤5: 自动清理临时文件...")
            self.update_progress(100)

            # 在视频处理完成后，自动清理所有临时文件，只保留处理好的视频
            if not self.test_mode_var.get():
                # 检查是否成功生成了最终视频
                output_filename = f"{self.video_base_name}_super_resolved.mp4"
                final_output = os.path.join(self.output_dir.get(), output_filename)

                if os.path.exists(final_output):
                    # 清理除04_processed_segments和最终输出外的所有临时文件
                    self.cleanup_temp_after_success(temp_dirs)
                else:
                    self.log("最终视频未生成，保留临时文件")
            else:
                self.log("测试模式：保留临时文件供检查")

            # 视频处理完成统计
            total_video_time = time.time() - video_start_time

            # 计算详细时间统计
            if self.enable_dup_detect_var.get() and not self.test_mode_var.get():
                total_dup_time = 0  # 这里需要从片段处理中累计
                total_sr_time = total_segment_time - total_dup_time  # 估算
            else:
                total_dup_time = 0
                total_sr_time = total_segment_time

            avg_frame_time = total_segment_time / max(total_frames_processed, 1) if total_frames_processed > 0 else 0

            self.log("=" * 60)
            self.log("视频处理完成详细统计:")
            self.log(f"  视频名称: {os.path.basename(video_path)}")
            self.log(f"  总处理时间: {total_video_time:.2f}秒")

            if not self.test_mode_var.get():
                self.log(f"  模型加载时间: {model_load_time if 'model_load_time' in locals() else 0:.2f}秒")
                self.log(f"  视频分割时间: {split_time if 'split_time' in locals() else 0:.2f}秒")
                self.log(f"  片段处理总时间: {total_segment_time:.2f}秒")

                if self.enable_dup_detect_var.get():
                    self.log(f"    重复帧检测时间: {total_dup_time:.2f}秒")
                    self.log(f"    超分辨率处理时间: {total_sr_time:.2f}秒")
                    self.log(f"  总计重复帧: {self.dup_frame_count}个")

                if total_frames_processed > 0:
                    self.log(f"  处理总帧数: {total_frames_processed}")
                    self.log(f"  平均每帧处理时间: {avg_frame_time:.3f}秒")

                    # 计算处理速度
                    processing_speed = total_frames_processed / total_segment_time if total_segment_time > 0 else 0
                    self.log(f"  处理速度: {processing_speed:.1f} 帧/秒")
            else:
                self.log("测试模式统计:")
                self.log(f"  总检测时间: {total_video_time:.2f}秒")
                self.log(f"  总计检测到重复帧: {self.dup_frame_count}个")
                self.log(f"  检测总帧数: {total_frames_processed}")
                if total_frames_processed > 0:
                    avg_detection_time = total_segment_time / total_frames_processed
                    self.log(f"  平均每帧检测时间: {avg_detection_time:.3f}秒")

            self.log("=" * 60)

            if self.test_mode_var.get():
                self.log(f"测试模式完成！测试结果保存在: {temp_dirs['base']}")
            else:
                self.log(f"处理完成！输出文件: {self.video_base_name}_super_resolved.mp4")

            # 重置状态
            self.current_segment_index = 0
            self.current_frame_in_segment = 0
            self.total_segments = 0
            self.segments = []
            self.processed_segments = []
            self.dup_frame_count = 0
            self.update_dup_info(0)

            # 清空历史缓存
            self.clear_history_cache()

            return True

        except Exception as e:
            self.log(f"处理视频失败: {str(e)}")
            import traceback
            self.log(f"错误详情:\n{traceback.format_exc()}")
            return False

    def cleanup_temp_after_success(self, temp_dirs):
        """成功处理后自动清理临时文件"""
        try:
            # 只保留04_processed_segments目录，清理其他临时目录
            dirs_to_clean = [
                temp_dirs['original_segments'],
                temp_dirs['audio'],
                temp_dirs['segment_frames'],
                os.path.join(temp_dirs['base'], "05_immediate_merge")
            ]

            cleaned_count = 0
            for dir_path in dirs_to_clean:
                if os.path.exists(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                        cleaned_count += 1
                        self.log(f"已清理临时目录: {os.path.basename(dir_path)}")
                    except Exception as e:
                        self.log(f"清理临时目录时出错({os.path.basename(dir_path)}): {e}")

            self.log(f"自动清理完成，共清理 {cleaned_count} 个临时目录")
        except Exception as e:
            self.log(f"自动清理临时文件时出错: {e}")

    def concatenate_videos(self, video_list, output_path):
        """拼接视频片段"""
        self.log("开始拼接视频片段...")
        start_time = time.time()

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
            concat_start = time.time()
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            concat_time = time.time() - concat_start

            total_time = time.time() - start_time
            self.log(f"视频拼接完成: {output_path}")
            self.log(f"  总耗时: {total_time:.2f}秒 (拼接: {concat_time:.2f}s)")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"视频拼接失败: {e.stderr}")
            raise
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)

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

            # 验证参数（添加空值检查）
            try:
                hash_threshold_str = self.hash_threshold_var.get()
                ssim_threshold_str = self.ssim_threshold_var.get()
                history_size_str = self.history_size_var.get()

                # 处理空值
                hash_threshold = int(hash_threshold_str) if hash_threshold_str else 3
                ssim_threshold = float(ssim_threshold_str) if ssim_threshold_str else 0.98
                history_size = int(history_size_str) if history_size_str else 20

                if hash_threshold < 0 or hash_threshold > 10:
                    messagebox.showwarning("警告", "哈希相似度阈值必须在0-10之间")
                    self.hash_threshold_var.set("3")
                    return

                if ssim_threshold < 0.9 or ssim_threshold > 1.0:
                    messagebox.showwarning("警告", "SSIM阈值必须在0.9-1.0之间")
                    self.ssim_threshold_var.set("0.98")
                    return

                if history_size < 1 or history_size > 200:
                    messagebox.showwarning("警告", "历史帧数量必须在1-200之间")
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

            # 根据处理模式设置暂停按钮状态
            if not self.enable_dup_detect_var.get():
                self.pause_btn.config(state='disabled')  # 直接处理模式禁用暂停
                self.log("直接处理模式：暂停功能已禁用")
            else:
                self.pause_btn.config(state='normal')  # 逐帧处理模式启用暂停

            self.stop_btn.config(state='normal')
            self.update_status("批量处理中...", "blue")

            # 启动内存监控
            self.start_memory_monitor()

            # 处理每个视频
            total_videos = len(self.input_paths)
            total_start_time = time.time()

            self.log("=" * 60)
            self.log(f"开始批量处理 {total_videos} 个视频")
            self.log("=" * 60)

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
                self.log(f"批量处理已停止")
                self.update_status("已停止", "orange")
                return

            # 所有视频处理完成
            total_elapsed = time.time() - total_start_time

            self.log("=" * 60)
            self.log("批量处理完成统计:")
            self.log(f"  处理视频总数: {total_videos}")
            self.log(f"  总处理时间: {total_elapsed:.2f}秒")
            self.log(f"  平均每个视频处理时间: {total_elapsed / total_videos:.2f}秒")
            self.log("=" * 60)

            if self.test_mode_var.get():
                self.update_status("测试完成！", "green")
                messagebox.showinfo("测试完成",
                                    f"批量测试完成！\n\n"
                                    f"共处理 {total_videos} 个视频\n"
                                    f"总耗时: {total_elapsed:.2f}秒\n"
                                    f"测试结果保存在各视频的临时目录中")
            else:
                self.update_status("批量处理完成！", "green")
                messagebox.showinfo("完成",
                                    f"批量处理完成！\n\n"
                                    f"共处理 {total_videos} 个视频\n"
                                    f"总耗时: {total_elapsed:.2f}秒\n"
                                    f"输出目录: {output_dir}")

            # 重置状态
            self.current_video_index = 0

        except Exception as e:
            self.log(f"批量处理失败: {str(e)}")
            self.update_status(f"处理失败: {str(e)}", "red")
            messagebox.showerror("错误", f"处理失败: {str(e)}")
        finally:
            self.processing = False
            self.paused = False
            self.stopped = False
            self.process_btn.config(state='normal')
            self.pause_btn.config(state='disabled', text="⏸ 暂停")
            self.stop_btn.config(state='disabled')

            # 清理GPU内存
            if self.generator and not self.test_mode_var.get():
                try:
                    # 确保模型从GPU移除
                    self.generator = self.generator.cpu()
                    del self.generator
                    self.generator = None
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # 等待CUDA操作完成
                    self.log("GPU内存已完全释放")
                except Exception as e:
                    self.log(f"清理GPU内存时出错: {e}")

    def start_processing(self):
        """开始处理"""
        if self.processing:
            return

        # 验证参数
        try:
            scale = int(self.scale_var.get())
            model = self.model_var.get()
            history_size = int(self.history_size_var.get())

            if model in ["GRL", "DAT"] and scale != 4:
                messagebox.showwarning("警告", f"{model}模型只支持4倍缩放")
                self.scale_var.set("4")
                return

            if scale not in [2, 4]:
                messagebox.showwarning("警告", "缩放因子必须是2或4")
                return

            if history_size < 1 or history_size > 200:
                messagebox.showwarning("警告", "历史帧数量必须在1-200之间")
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

        # 直接处理模式不支持暂停
        if not self.enable_dup_detect_var.get():
            messagebox.showinfo("提示", "直接处理模式不支持暂停功能")
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
            self.log("处理暂停")

    def stop_processing(self):
        """停止处理"""
        if not self.processing:
            return

        response = messagebox.askyesno("停止处理",
                                       "是否确认停止处理？")

        if not response:
            return

        self.log("正在停止处理...")
        self.update_status("正在停止...", "orange")
        self.stopped = True
        self.paused = False  # 确保暂停状态被清除

        # 通知暂停的线程继续（如果是暂停状态）
        with self.pause_cv:
            self.pause_cv.notify_all()

        # 等待处理线程响应
        time.sleep(0.5)

        self.log("处理已停止")
        self.update_status("已停止", "orange")

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
            response = messagebox.askyesno("退出",
                                           "处理仍在进行中，是否确认退出？\n\n"
                                           "退出后进度将不会保存，下次需要重新开始处理。")

            if not response:
                return

            self.log("正在停止处理并退出...")
            self.processing = False
            self.stopped = True
            self.paused = False

            # 通知暂停的线程继续（如果是暂停状态）
            with self.pause_cv:
                self.pause_cv.notify_all()

            time.sleep(1.0)  # 给线程更多时间响应

        self.root.destroy()


def main():
    """主函数"""
    app = APISRVideoProcessor()
    app.run()


if __name__ == "__main__":
    main()