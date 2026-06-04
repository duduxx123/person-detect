import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class PersonTrackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("人员追踪控制台")
        self.root.geometry("1280x720")
        self.root.minsize(1000, 620)

        self.colors = {
            "bg": "#091017",
            "panel": "#101923",
            "panel_alt": "#0c141d",
            "border": "#1b2c3a",
            "border_bright": "#29506b",
            "text": "#e6eef6",
            "muted": "#8aa0b7",
            "accent": "#2de0c2",
            "accent_soft": "#143a37",
            "success": "#39d98a",
            "warning": "#f5b971",
            "danger": "#ff6b81",
            "entry": "#0d151e",
            "terminal": "#04080d",
            "terminal_text": "#d9fff4",
            "grid": "#0f2430",
            "scanline": "#0d1d24",
        }

        self.base_dir = Path(__file__).resolve().parent
        self.script_path = self.base_dir / "person_track.py"
        self.logs_dir = self.base_dir / "logs"
        self.process = None
        self.process_start_time = None
        self.log_queue = queue.Queue()
        self.log_buffer = []
        self.log_scrollbar_view = (0.0, 1.0)
        self.log_scrollbar_thumb = (0, 0)
        self.log_scrollbar_drag_offset = 0

        self.source_var = tk.StringVar(value="0")
        self.model_var = tk.StringVar(value=str(self.base_dir / "yolo11n.pt"))
        self.tracker_var = tk.StringVar(value=str(self.base_dir / "custom_person_botsort.yaml"))
        self.reid_var = tk.StringVar(
            value=str(
                self.base_dir
                / "osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
            )
        )
        self.window_title_var = tk.StringVar(value="YOLO11 人员追踪")
        self.status_var = tk.StringVar(value="系统待命")
        self.detail_var = tk.StringVar(value="等待启动检测任务")
        self.pid_var = tk.StringVar(value="PID: -")
        self.runtime_var = tk.StringVar(value="运行时长: 0s")
        self.source_info_var = tk.StringVar(value="视频源: 0")
        self.model_info_var = tk.StringVar(value="模型: yolo11n.pt")
        self.advanced_visible = tk.BooleanVar(value=False)
        self.indicator_state = False

        self._configure_root()
        self._configure_styles()
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.poll_log_queue)
        self.root.after(500, self.poll_process_state)

    def _configure_root(self):
        self.root.configure(bg=self.colors["bg"])

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=self.colors["bg"], foreground=self.colors["text"])
        style.configure("App.TFrame", background=self.colors["bg"])
        style.configure("Panel.TFrame", background=self.colors["panel"])
        style.configure("HeaderTitle.TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Microsoft YaHei UI", 22, "bold"))
        style.configure("HeaderSub.TLabel", background=self.colors["bg"], foreground=self.colors["muted"], font=("Consolas", 10))
        style.configure(
            "Section.TLabelframe",
            background=self.colors["panel"],
            borderwidth=1,
            relief="solid",
            bordercolor=self.colors["border"],
        )
        style.configure(
            "Section.TLabelframe.Label",
            background=self.colors["panel"],
            foreground=self.colors["accent"],
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure("Field.TLabel", background=self.colors["panel"], foreground=self.colors["text"], font=("Microsoft YaHei UI", 10))
        style.configure("Hint.TLabel", background=self.colors["panel"], foreground=self.colors["muted"], font=("Microsoft YaHei UI", 9))
        style.configure(
            "Dark.TEntry",
            fieldbackground=self.colors["entry"],
            background=self.colors["entry"],
            foreground=self.colors["text"],
            insertcolor=self.colors["accent"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
            padding=8,
        )
        style.map("Dark.TEntry", fieldbackground=[("disabled", "#0a1017")], foreground=[("disabled", "#6c8398")])
        style.configure("Accent.TButton", background=self.colors["accent"], foreground="#071015", font=("Microsoft YaHei UI", 10, "bold"), borderwidth=0, padding=(16, 10))
        style.map("Accent.TButton", background=[("active", "#55efd6"), ("disabled", "#23433d")], foreground=[("disabled", "#7da79d")])
        style.configure("Secondary.TButton", background="#172432", foreground=self.colors["text"], font=("Microsoft YaHei UI", 10, "bold"), borderwidth=0, padding=(16, 10))
        style.map("Secondary.TButton", background=[("active", "#21364a"), ("disabled", "#101922")], foreground=[("disabled", "#617281")])
        style.configure("Danger.TButton", background="#411922", foreground="#ffe2e8", font=("Microsoft YaHei UI", 10, "bold"), borderwidth=0, padding=(16, 10))
        style.map("Danger.TButton", background=[("active", "#5d2630"), ("disabled", "#221116")], foreground=[("disabled", "#8f6871")])
        style.configure(
            "Terminal.Vertical.TScrollbar",
            background=self.colors["border_bright"],
            troughcolor=self.colors["terminal"],
            bordercolor=self.colors["terminal"],
            arrowcolor=self.colors["accent"],
            lightcolor=self.colors["border_bright"],
            darkcolor=self.colors["border"],
        )

    def _build_ui(self):
        container = ttk.Frame(self.root, style="App.TFrame", padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=0)
        container.rowconfigure(2, weight=1)

        self._build_header(container)

        top_area = tk.Frame(container, bg=self.colors["bg"])
        top_area.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        top_area.columnconfigure(0, weight=3)
        top_area.columnconfigure(1, weight=2)
        self.top_area = top_area
        self._build_config_panel(top_area)
        self._build_side_panel(top_area)

        bottom_area = tk.Frame(container, bg=self.colors["bg"], height=118)
        bottom_area.grid(row=2, column=0, sticky="nsew")
        bottom_area.grid_propagate(False)
        bottom_area.rowconfigure(0, weight=1)
        bottom_area.columnconfigure(0, weight=1)
        self.bottom_area = bottom_area
        self._build_log_panel(bottom_area)

        self.append_log("控制台已就绪，等待任务启动。", "success")

    def _build_header(self, parent):
        header = ttk.Frame(parent, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(1, weight=1)

        indicator_wrap = tk.Frame(header, bg=self.colors["bg"])
        indicator_wrap.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 14))

        self.indicator_canvas = tk.Canvas(
            indicator_wrap,
            width=30,
            height=30,
            bg=self.colors["bg"],
            highlightthickness=0,
            bd=0,
        )
        self.indicator_canvas.pack(side="left")
        self.indicator_text = tk.Label(
            indicator_wrap,
            text="离线",
            bg=self.colors["bg"],
            fg=self.colors["danger"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.indicator_text.pack(side="left", padx=(8, 0))

        ttk.Label(header, text="人员追踪控制台", style="HeaderTitle.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(header, text="YOLO11 + ReID 深色终端监控面板", style="HeaderSub.TLabel").grid(row=1, column=1, sticky="w", pady=(4, 0))
        self._render_indicator(False)

    def _build_config_panel(self, parent):
        outer = self._create_terminal_panel(parent, row=0, column=0, title="基础配置", padx=(0, 10))
        outer.columnconfigure(0, weight=1)

        content = tk.Frame(outer, bg=self.colors["panel"])
        content.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        content.columnconfigure(1, weight=1)

        self.source_entry = self._add_row(content, 0, "视频源", self.source_var, "可填摄像头编号、视频文件路径或流地址")
        self.model_entry = self._add_row(
            content,
            1,
            "YOLO 模型",
            self.model_var,
            None,
            lambda: self.pick_file(self.model_var, [("PyTorch 模型", "*.pt"), ("所有文件", "*.*")]),
        )
        self.tracker_entry = self._add_row(
            content,
            2,
            "Tracker 配置",
            self.tracker_var,
            None,
            lambda: self.pick_file(self.tracker_var, [("YAML 配置", "*.yaml"), ("所有文件", "*.*")]),
        )
        self.reid_entry = self._add_row(
            content,
            3,
            "ReID 权重",
            self.reid_var,
            None,
            lambda: self.pick_file(self.reid_var, [("PyTorch 权重", "*.pth"), ("所有文件", "*.*")]),
        )

        advanced_toggle = ttk.Button(content, text="展开高级配置", style="Secondary.TButton", command=self.toggle_advanced)
        advanced_toggle.grid(row=8, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.advanced_toggle_button = advanced_toggle

        self.advanced_frame = tk.Frame(content, bg=self.colors["panel_alt"], highlightbackground=self.colors["border"], highlightthickness=1, bd=0)
        self.advanced_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        self.advanced_frame.columnconfigure(1, weight=1)

        self.window_title_entry = self._add_row(
            self.advanced_frame,
            0,
            "窗口标题",
            self.window_title_var,
            "用于 OpenCV 预览窗口的标题显示",
        )

        self.advanced_hint = tk.Label(
            self.advanced_frame,
            text="高级配置区域可继续扩展更多阈值或运行参数。",
            bg=self.colors["panel_alt"],
            fg=self.colors["muted"],
            font=("Microsoft YaHei UI", 9),
            anchor="w",
        )
        self.advanced_hint.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=(2, 12))
        self.advanced_frame.grid_remove()

    def _build_side_panel(self, parent):
        side = tk.Frame(parent, bg=self.colors["bg"])
        side.grid(row=0, column=1, sticky="nsew")
        side.rowconfigure(1, weight=1)

        control_panel = self._create_terminal_panel(side, row=0, column=0, title="任务控制")
        button_row = tk.Frame(control_panel, bg=self.colors["panel"])
        button_row.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        button_row.columnconfigure((0, 1, 2), weight=1)

        self.start_button = ttk.Button(button_row, text="开始检测", style="Accent.TButton", command=self.start_detection)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.stop_button = ttk.Button(button_row, text="停止检测", style="Danger.TButton", command=self.stop_detection, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self.save_log_button = ttk.Button(button_row, text="导出日志", style="Secondary.TButton", command=self.save_logs)
        self.save_log_button.grid(row=0, column=2, sticky="ew")

        telemetry_panel = self._create_terminal_panel(side, row=1, column=0, title="运行遥测", pady=(10, 0))
        self.telemetry_body = tk.Frame(telemetry_panel, bg=self.colors["panel"])
        self.telemetry_body.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.telemetry_body.columnconfigure(0, weight=1)

        self.status_label = tk.Label(
            self.telemetry_body,
            textvariable=self.status_var,
            bg=self.colors["panel_alt"],
            fg=self.colors["text"],
            font=("Microsoft YaHei UI", 14, "bold"),
            anchor="w",
            padx=14,
            pady=12,
        )
        self.status_label.grid(row=0, column=0, sticky="ew")

        self.detail_label = tk.Label(
            self.telemetry_body,
            textvariable=self.detail_var,
            bg=self.colors["panel_alt"],
            fg=self.colors["muted"],
            font=("Microsoft YaHei UI", 10),
            anchor="w",
            justify="left",
            padx=14,
            pady=10,
        )
        self.detail_label.grid(row=1, column=0, sticky="ew", pady=(8, 10))

        self._create_metric_card(self.telemetry_body, 2, self.pid_var)
        self._create_metric_card(self.telemetry_body, 3, self.runtime_var)
        self._create_metric_card(self.telemetry_body, 4, self.source_info_var)
        self._create_metric_card(self.telemetry_body, 5, self.model_info_var)

    def _build_log_panel(self, parent):
        log_panel = self._create_terminal_panel(parent, row=0, column=0, title="事件日志流")

        self.log_container = tk.Frame(
            log_panel,
            bg=self.colors["terminal"],
            highlightbackground=self.colors["border_bright"],
            highlightthickness=1,
            bd=0,
        )
        self.log_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.log_container.rowconfigure(0, weight=1)
        self.log_container.columnconfigure(0, weight=1)
        self.log_container.columnconfigure(1, weight=0)

        self._draw_scanline_background(self.log_container)

        self.log_text = tk.Text(
            self.log_container,
            height=3,
            wrap="word",
            font=("Consolas", 11),
            bg=self.colors["terminal"],
            fg=self.colors["terminal_text"],
            insertbackground=self.colors["accent"],
            selectbackground="#103f3d",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
        )
        self.log_scrollbar = tk.Canvas(
            self.log_container,
            width=18,
            height=1,
            bg=self.colors["terminal"],
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.log_text.configure(yscrollcommand=self._sync_log_scrollbar)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=6)
        self.log_scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)
        self.log_scrollbar.bind("<Configure>", lambda _event: self._redraw_log_scrollbar())
        self.log_scrollbar.bind("<Button-1>", self._on_log_scrollbar_click)
        self.log_scrollbar.bind("<B1-Motion>", self._on_log_scrollbar_drag)
        self.log_scrollbar.bind("<MouseWheel>", self._on_log_mousewheel)
        self.log_text.bind("<MouseWheel>", self._on_log_mousewheel)
        self.log_text.bind("<Button-4>", lambda _event: self._scroll_log_units(-3))
        self.log_text.bind("<Button-5>", lambda _event: self._scroll_log_units(3))
        self.log_text.configure(state="disabled")

        self.log_text.tag_configure("default", foreground=self.colors["terminal_text"])
        self.log_text.tag_configure("success", foreground=self.colors["success"])
        self.log_text.tag_configure("error", foreground=self.colors["danger"])
        self.log_text.tag_configure("warning", foreground=self.colors["warning"])
        self.log_text.tag_configure("info", foreground=self.colors["accent"])
        self.log_text.tag_configure("muted", foreground=self.colors["muted"])

    def _sync_log_scrollbar(self, first, last):
        self.log_scrollbar_view = (float(first), float(last))
        self._redraw_log_scrollbar()

    def _redraw_log_scrollbar(self):
        if not hasattr(self, "log_scrollbar"):
            return

        width = max(self.log_scrollbar.winfo_width(), 18)
        height = max(self.log_scrollbar.winfo_height(), 1)
        first, last = self.log_scrollbar_view
        self.log_scrollbar.delete("all")

        track_x1 = 5
        track_x2 = width - 5
        self.log_scrollbar.create_rectangle(
            track_x1,
            0,
            track_x2,
            height,
            fill=self.colors["border_bright"],
            outline=self.colors["border"],
        )

        thumb_top = int(first * height)
        thumb_bottom = int(last * height)
        min_thumb_height = min(34, max(height, 1))
        if thumb_bottom - thumb_top < min_thumb_height:
            center = (thumb_top + thumb_bottom) // 2
            thumb_top = max(0, center - min_thumb_height // 2)
            thumb_bottom = min(height, thumb_top + min_thumb_height)
            thumb_top = max(0, thumb_bottom - min_thumb_height)

        self.log_scrollbar_thumb = (thumb_top, thumb_bottom)
        self.log_scrollbar.create_rectangle(
            3,
            thumb_top,
            width - 3,
            thumb_bottom,
            fill=self.colors["accent"],
            outline="#83ffea",
        )
        grip_y = (thumb_top + thumb_bottom) // 2
        self.log_scrollbar.create_line(6, grip_y - 4, width - 6, grip_y - 4, fill=self.colors["terminal"])
        self.log_scrollbar.create_line(6, grip_y, width - 6, grip_y, fill=self.colors["terminal"])
        self.log_scrollbar.create_line(6, grip_y + 4, width - 6, grip_y + 4, fill=self.colors["terminal"])

    def _on_log_scrollbar_click(self, event):
        thumb_top, thumb_bottom = self.log_scrollbar_thumb
        if thumb_top <= event.y <= thumb_bottom:
            self.log_scrollbar_drag_offset = event.y - thumb_top
        else:
            thumb_height = max(thumb_bottom - thumb_top, 1)
            self.log_scrollbar_drag_offset = thumb_height // 2
            self._move_log_scrollbar_to(event.y - self.log_scrollbar_drag_offset)

    def _on_log_scrollbar_drag(self, event):
        self._move_log_scrollbar_to(event.y - self.log_scrollbar_drag_offset)

    def _move_log_scrollbar_to(self, thumb_top):
        height = max(self.log_scrollbar.winfo_height(), 1)
        current_top, current_bottom = self.log_scrollbar_thumb
        thumb_height = max(current_bottom - current_top, 1)
        travel = max(height - thumb_height, 1)
        clamped_top = min(max(thumb_top, 0), travel)
        self.log_text.yview_moveto(clamped_top / travel)

    def _on_log_mousewheel(self, event):
        units = -1 if event.delta > 0 else 1
        self._scroll_log_units(units * 3)
        return "break"

    def _scroll_log_units(self, units):
        self.log_text.yview_scroll(units, "units")
        return "break"

    def _create_terminal_panel(self, parent, row, column, title, padx=(0, 0), pady=(0, 0)):
        outer = tk.Frame(parent, bg=self.colors["bg"])
        outer.grid(row=row, column=column, sticky="nsew", padx=padx, pady=pady)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        frame = tk.Frame(
            outer,
            bg=self.colors["panel"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
        )
        frame.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        self._draw_grid_texture(frame)

        title_bar = tk.Frame(frame, bg=self.colors["panel_alt"], height=34)
        title_bar.grid(row=0, column=0, sticky="ew")
        title_bar.grid_propagate(False)

        tk.Label(
            title_bar,
            text=title,
            bg=self.colors["panel_alt"],
            fg=self.colors["accent"],
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor="w",
            padx=14,
        ).pack(fill="both", expand=True)

        return frame

    def _draw_grid_texture(self, parent):
        canvas = tk.Canvas(parent, bg=self.colors["panel"], highlightthickness=0, bd=0)
        canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        canvas.tk.call("lower", canvas._w)

        def redraw(event):
            canvas.delete("grid")
            width = max(event.width, 1)
            height = max(event.height, 1)
            step = 24
            for x in range(0, width, step):
                canvas.create_line(x, 0, x, height, fill=self.colors["grid"], tags="grid")
            for y in range(0, height, step):
                canvas.create_line(0, y, width, y, fill=self.colors["grid"], tags="grid")

        canvas.bind("<Configure>", redraw)

    def _draw_scanline_background(self, parent):
        canvas = tk.Canvas(parent, bg=self.colors["terminal"], highlightthickness=0, bd=0)
        canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        canvas.tk.call("lower", canvas._w)

        def redraw(event):
            canvas.delete("scan")
            width = max(event.width, 1)
            height = max(event.height, 1)
            for y in range(0, height, 4):
                canvas.create_line(0, y, width, y, fill=self.colors["scanline"], tags="scan")

        canvas.bind("<Configure>", redraw)

    def _create_metric_card(self, parent, row, variable):
        card = tk.Frame(parent, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1, bd=0)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        tk.Label(
            card,
            textvariable=variable,
            bg=self.colors["panel"],
            fg=self.colors["accent"],
            font=("Consolas", 11),
            anchor="w",
            padx=12,
            pady=10,
        ).pack(fill="x")

    def _add_row(self, parent, row, label, variable, helper_text=None, browse_command=None):
        base_row = row * 2
        bg = parent.cget("bg")

        tk.Label(parent, text=label, bg=bg, fg=self.colors["text"], font=("Microsoft YaHei UI", 10)).grid(
            row=base_row, column=0, sticky="w", padx=12, pady=(12, 0)
        )
        entry = ttk.Entry(parent, textvariable=variable, style="Dark.TEntry")
        entry.grid(row=base_row, column=1, sticky="ew", padx=(12, 12), pady=(12, 0))

        if browse_command is not None:
            ttk.Button(parent, text="选择文件", style="Secondary.TButton", command=browse_command).grid(
                row=base_row, column=2, sticky="ew", padx=(0, 12), pady=(12, 0)
            )

        if helper_text:
            tk.Label(parent, text=helper_text, bg=bg, fg=self.colors["muted"], font=("Microsoft YaHei UI", 9)).grid(
                row=base_row + 1, column=1, columnspan=2, sticky="w", padx=(12, 12), pady=(4, 0)
            )
        else:
            tk.Frame(parent, bg=bg, height=10).grid(row=base_row + 1, column=0, columnspan=3)

        return entry

    def toggle_advanced(self):
        visible = self.advanced_visible.get()
        if visible:
            self.advanced_frame.grid_remove()
            self.advanced_toggle_button.configure(text="展开高级配置")
            self.advanced_visible.set(False)
        else:
            self.advanced_frame.grid()
            self.advanced_toggle_button.configure(text="收起高级配置")
            self.advanced_visible.set(True)

    def _render_indicator(self, online):
        self.indicator_canvas.delete("all")
        glow = self.colors["success"] if online else self.colors["danger"]
        solid = "#7dffd2" if online else "#ff9dad"
        for radius, stipple in ((22, "gray50"), (16, "gray25")):
            self.indicator_canvas.create_oval(
                15 - radius / 2,
                15 - radius / 2,
                15 + radius / 2,
                15 + radius / 2,
                fill=glow,
                outline="",
                stipple=stipple,
            )
        self.indicator_canvas.create_oval(9, 9, 21, 21, fill=solid, outline="")
        self.indicator_text.configure(text="在线" if online else "离线", fg=glow)
        self.indicator_state = online

    def pick_file(self, target_var, filetypes):
        selected = filedialog.askopenfilename(initialdir=self.base_dir, filetypes=filetypes)
        if selected:
            target_var.set(selected)

    def classify_log_tag(self, message):
        lower = message.lower()
        if any(word in lower for word in ("error", "traceback", "cannot", "failed", "异常", "失败", "不存在")):
            return "error"
        if any(word in lower for word in ("start", "started", "online", "成功", "启动", "已启动", "已就绪")):
            return "success"
        if any(word in lower for word in ("stop", "exited", "warning", "停止", "退出", "等待")):
            return "warning"
        if any(word in lower for word in ("pid", "source", "model", "reid", "tracker")):
            return "info"
        return "default"

    def append_log(self, message, tag=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_buffer.append(line)
        resolved_tag = tag or self.classify_log_tag(message)
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{line}\n", resolved_tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def start_detection(self):
        if self.process and self.process.poll() is None:
            self.append_log("检测已经在运行中。", "warning")
            return

        if not self.script_path.exists():
            messagebox.showerror("文件缺失", f"未找到脚本：\n{self.script_path}")
            return

        source = self.source_var.get().strip()
        model_path = Path(self.model_var.get().strip())
        tracker_path = Path(self.tracker_var.get().strip())
        reid_path = Path(self.reid_var.get().strip())
        window_title = self.window_title_var.get().strip() or "YOLO11 人员追踪"

        if not source:
            messagebox.showwarning("参数错误", "请填写视频源。")
            return

        missing_paths = [path for path in (model_path, tracker_path, reid_path) if not path.exists()]
        if missing_paths:
            messagebox.showerror("文件缺失", "以下文件不存在：\n" + "\n".join(str(path) for path in missing_paths))
            return

        command = [
            sys.executable,
            str(self.script_path),
            "--source",
            source,
            "--model",
            str(model_path),
            "--tracker",
            str(tracker_path),
            "--reid-weights",
            str(reid_path),
            "--window-title",
            window_title,
        ]
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except Exception as exc:
            messagebox.showerror("启动失败", f"无法启动检测进程：\n{exc}")
            return

        self.process_start_time = time.time()
        threading.Thread(target=self._read_process_output, daemon=True).start()
        self._set_running_state()
        self.append_log(
            f"检测进程已启动，PID={self.process.pid}，视频源={source}，模型={model_path.name}，ReID={reid_path.name}",
            "success",
        )

    def stop_detection(self):
        if not self.process or self.process.poll() is not None:
            self.append_log("当前没有正在运行的检测进程。", "warning")
            self._set_idle_state()
            return

        self.status_var.set("正在停止")
        self.detail_var.set("已发送停止信号，等待检测进程安全退出...")
        self.append_log("正在停止检测进程...", "warning")
        try:
            if sys.platform.startswith("win"):
                try:
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:
                    self.process.terminate()
            else:
                self.process.terminate()
        except Exception as exc:
            self.append_log(f"停止进程时出现异常：{exc}", "error")

        self.root.after(1800, self._force_stop_if_needed)

    def _force_stop_if_needed(self):
        if self.process and self.process.poll() is None:
            self.append_log("检测进程仍未退出，执行强制结束。", "error")
            try:
                self.process.kill()
            except Exception as exc:
                self.append_log(f"强制结束失败：{exc}", "error")

    def _read_process_output(self):
        if not self.process or not self.process.stdout:
            return

        for line in self.process.stdout:
            self.log_queue.put(line.rstrip())

        return_code = self.process.wait()
        self.log_queue.put(("Process exited with code=" + str(return_code)))

    def poll_log_queue(self):
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                if message:
                    self.append_log(message)
        self.root.after(100, self.poll_log_queue)

    def poll_process_state(self):
        if self.process and self.process.poll() is None:
            elapsed = int(time.time() - self.process_start_time) if self.process_start_time else 0
            model_name = Path(self.model_var.get()).name if self.model_var.get().strip() else "-"
            source = self.source_var.get().strip() or "-"
            self.status_var.set("在线运行中")
            self.detail_var.set("YOLO11 检测与 ReID 追踪链路正在持续处理视频帧。")
            self.pid_var.set(f"PID: {self.process.pid}")
            self.runtime_var.set(f"运行时长: {elapsed}s")
            self.source_info_var.set(f"视频源: {source}")
            self.model_info_var.set(f"模型: {model_name}")
            self._render_indicator(True)
            self._apply_status_theme(running=True)
        elif self.process and self.process.poll() is not None:
            self._set_idle_state()
        self.root.after(500, self.poll_process_state)

    def _apply_status_theme(self, running):
        if running:
            self.status_label.configure(bg=self.colors["accent_soft"], fg=self.colors["accent"])
            self.detail_label.configure(bg=self.colors["panel_alt"], fg=self.colors["muted"])
        else:
            self.status_label.configure(bg=self.colors["panel_alt"], fg=self.colors["text"])
            self.detail_label.configure(bg=self.colors["panel_alt"], fg=self.colors["muted"])

    def _set_running_state(self):
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.source_entry.configure(state="disabled")
        self.model_entry.configure(state="disabled")
        self.tracker_entry.configure(state="disabled")
        self.reid_entry.configure(state="disabled")
        self.window_title_entry.configure(state="disabled")
        self.status_var.set("启动中")
        self.detail_var.set("正在拉起检测进程并绑定日志通道...")
        self.source_info_var.set(f"视频源: {self.source_var.get().strip() or '-'}")
        self.model_info_var.set(f"模型: {Path(self.model_var.get()).name if self.model_var.get().strip() else '-'}")
        self._render_indicator(True)
        self._apply_status_theme(running=True)

    def _set_idle_state(self):
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.source_entry.configure(state="normal")
        self.model_entry.configure(state="normal")
        self.tracker_entry.configure(state="normal")
        self.reid_entry.configure(state="normal")
        self.window_title_entry.configure(state="normal")
        self.status_var.set("系统待命")
        self.detail_var.set("等待启动检测任务")
        self.pid_var.set("PID: -")
        self.runtime_var.set("运行时长: 0s")
        self.source_info_var.set(f"视频源: {self.source_var.get().strip() or '-'}")
        self.model_info_var.set(f"模型: {Path(self.model_var.get()).name if self.model_var.get().strip() else '-'}")
        self.process = None
        self.process_start_time = None
        self._render_indicator(False)
        self._apply_status_theme(running=False)

    def save_logs(self):
        if not self.log_buffer:
            messagebox.showinfo("没有日志", "当前还没有日志可导出。")
            return

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"person_track_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        target = filedialog.asksaveasfilename(
            title="导出日志",
            initialdir=self.logs_dir,
            initialfile=default_name,
            defaultextension=".log",
            filetypes=[("日志文件", "*.log"), ("文本文件", "*.txt"), ("所有文件", "*.*")],
        )
        if not target:
            return

        target_path = Path(target)
        target_path.write_text("\n".join(self.log_buffer) + "\n", encoding="utf-8")
        self.append_log(f"日志已导出到：{target_path}", "info")

    def on_close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("退出确认", "检测仍在运行，是否先停止检测并关闭控制台？"):
                return
            self.stop_detection()
            self.root.after(800, self.root.destroy)
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    PersonTrackGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
