import sys
import os
import numpy as np
import imageio
import cv2
import qdarkstyle
from skimage import io
from pathlib import Path
from natsort import natsorted
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QComboBox, QProgressBar, 
    QMessageBox, QGroupBox, QSlider, QFrame, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt5.QtGui import QPixmap, QIcon, QImage

IMAGE_EXTENSIONS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif')

def normalize_single_frame(img, global_max=None):
    """
    单帧归一化处理
    """
    if img.dtype == np.uint8:
        return img.copy()
    
    img = img.astype(np.float32)
    
    # 如果没提供全局最大值，就用当前帧最大值（懒加载模式下通常如此）
    # 如果提供了全局最大值（内存模式），则统一亮度
    max_val = global_max if global_max is not None else img.max()
    
    if max_val > 0:
        img = (img / max_val * 255.0)
    
    return np.clip(img, 0, 255).astype(np.uint8)

class ImgToVideoWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, data_source, out_path, fps, method, codec, quality, is_lazy, global_max=None):
        super().__init__()
        self.data_source = data_source # 可能是 np.array (内存模式) 也可能是 list (懒加载模式)
        self.out_path = Path(out_path)
        self.fps = fps
        self.method = method
        self.codec = codec
        self.quality = quality
        self.is_lazy = is_lazy
        self.global_max = global_max

    def get_frame(self, idx):
        """根据模式获取并归一化帧"""
        if self.is_lazy:
            # 懒加载：此时 data_source 是路径列表
            img_path = self.data_source[idx]
            frame = io.imread(img_path)
            return normalize_single_frame(frame, global_max=None)
        else:
            # 内存模式：此时 data_source 是 numpy 数组
            frame = self.data_source[idx]
            return normalize_single_frame(frame, global_max=self.global_max)

    def run(self):
        try:
            total = len(self.data_source)
            
            # 先读取第一帧来确定视频尺寸
            first_frame = self.get_frame(0)
            
            # 处理通道
            if first_frame.ndim == 2:
                H, W = first_frame.shape
                is_gray = True
            else:
                H, W, C = first_frame.shape
                is_gray = False

            # --- 初始化写入器 ---
            writer = None
            cv_out = None

            if self.method == 'imageio':
                ffmpeg_params = ['-pix_fmt', 'yuv420p']
                if self.codec == 'libx264':
                    ffmpeg_params = ['-crf', str(self.quality), '-pix_fmt', 'yuv420p']
                elif self.codec == 'mpeg4':
                    ffmpeg_params = ['-qscale:v', str(self.quality)] + ffmpeg_params

                writer = imageio.get_writer(
                    str(self.out_path), format='FFMPEG', mode='I', fps=self.fps,
                    codec=self.codec, input_params=['-s', f'{W}x{H}'],
                    ffmpeg_params=ffmpeg_params
                )
            else: # OpenCV
                fourcc_map = {'MJPG': 'MJPG', 'XVID': 'XVID', 'HFYU': 'HFYU'}
                fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(self.codec, 'MJPG'))
                cv_out = cv2.VideoWriter(str(self.out_path), fourcc, self.fps, (W, H))
                if not cv_out.isOpened():
                    raise RuntimeError("OpenCV VideoWriter 初始化失败")

            # --- 循环写入 ---
            for i in range(total):
                frame = self.get_frame(i)

                # 确保通道数正确 (VideoWriter 需要 3 通道)
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.ndim == 3 and frame.shape[-1] == 1:
                    frame = np.concatenate([frame] * 3, axis=-1)

                if self.method == 'imageio':
                    writer.append_data(frame)
                else:
                    # OpenCV 需要 BGR
                    cv_out.write(frame[..., ::-1])

                self.progress.emit(int((i + 1) / total * 100))

            # --- 清理 ---
            if writer: writer.close()
            if cv_out: cv_out.release()

            self.finished.emit(str(self.out_path))

        except Exception as e:
            self.error.emit(str(e))


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class ImgToVideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ImgtoVideo')
        self.setWindowIcon(QIcon(resource_path('IMGtoVideo.ico')))
        self.setStyleSheet("""
            QWidget { font-size: 14px; font-family: "Microsoft YaHei"; }
            QPushButton { padding: 6px; }
            QGroupBox { font-weight: bold; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; }
            QLabel#InfoLabel { color: #aaaaaa; font-style: italic; font-size: 12px; }
        """)
        self.resize(1000, 680)

        # --- 初始化设置 ---
        self.settings = QSettings("SciTools", "ImgToVideoConverter")
        
        # 【新增】用于跟踪当前加载的数据所在的目录，作为保存的默认参考
        self.current_source_dir = None

        # 播放定时器
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # --- 1. 输入输出 ---
        group_io = QGroupBox("1. 输入与输出")
        io_layout = QVBoxLayout()
        
        fl = QHBoxLayout()
        self.input_label = QLabel('未选择数据')
        btn_in = QPushButton('选择单图像(Multi-page)')
        btn_in.clicked.connect(self.select_img)
        btn_folder = QPushButton('选择文件夹 (Sequence)')
        btn_folder.clicked.connect(self.select_folder)
        fl.addWidget(btn_in); fl.addWidget(btn_folder); fl.addWidget(self.input_label)
        io_layout.addLayout(fl)
        
        # 内存优化开关
        self.chk_lazy = QCheckBox("启用内存优化 (动态加载)")
        self.chk_lazy.setToolTip("勾选后不将所有图片读入内存，而是边读边写。")
        io_layout.addWidget(self.chk_lazy)

        fo = QHBoxLayout()
        self.output_label = QLabel('未选择保存位置')
        btn_out = QPushButton('选择保存位置')
        btn_out.clicked.connect(self.select_output)
        fo.addWidget(btn_out); fo.addWidget(self.output_label)
        
        self.name_input = QLineEdit("output")
        self.name_input.setPlaceholderText("文件名")
        fo.addWidget(QLabel("文件名:"))
        fo.addWidget(self.name_input)
        io_layout.addLayout(fo)

        group_io.setLayout(io_layout)
        left_layout.addWidget(group_io)

        # --- 2. 参数设置 ---
        group_params = QGroupBox("2. 参数设置")
        param_layout = QVBoxLayout()
        
        pl = QHBoxLayout()
        pl.addWidget(QLabel('FPS:'))
        self.fps_input = QLineEdit('30')
        self.fps_input.setFixedWidth(50)
        pl.addWidget(self.fps_input)

        pl.addWidget(QLabel('方法:'))
        self.method_combo = QComboBox()
        self.method_combo.addItems(['imageio', 'opencv'])
        self.method_combo.currentTextChanged.connect(self.update_ui_logic)
        pl.addWidget(self.method_combo)

        pl.addWidget(QLabel('编码器:'))
        self.codec_combo = QComboBox()
        self.codec_combo.currentTextChanged.connect(self.update_ui_logic)
        pl.addWidget(self.codec_combo)

        pl.addWidget(QLabel('质量值:'))
        self.qual_input = QLineEdit('18')
        self.qual_input.setFixedWidth(50)
        pl.addWidget(self.qual_input)
        param_layout.addLayout(pl)

        # 说明文本 Label
        self.info_label = QLabel("说明信息")
        self.info_label.setObjectName("InfoLabel")
        self.info_label.setStyleSheet("font-size: 10pt;") 
        self.info_label.setWordWrap(True)
        param_layout.addWidget(self.info_label)

        group_params.setLayout(param_layout)
        left_layout.addWidget(group_params)

        # --- 3. 进度 ---
        group_convert = QGroupBox("3. 执行")
        cl = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        cl.addWidget(self.progress_bar)
        self.btn_convert = QPushButton('开始转换')
        self.btn_convert.clicked.connect(self.start_conversion)
        self.btn_convert.setMinimumHeight(40)
        cl.addWidget(self.btn_convert)
        group_convert.setLayout(cl)
        left_layout.addWidget(group_convert)

        # --- 右侧预览 ---
        preview_panel = QFrame()
        preview_panel.setFrameShape(QFrame.StyledPanel)
        preview_layout = QVBoxLayout()
        
        self.img_preview = QLabel("预览区域\n(加载后显示)")
        self.img_preview.setAlignment(Qt.AlignCenter)
        self.img_preview.setMinimumSize(640, 640)
        self.img_preview.setStyleSheet("background-color: #222; border: 1px solid #444;")
        preview_layout.addWidget(self.img_preview, 1)

        # 播放控制区域
        ctrl_layout = QHBoxLayout()
        
        self.btn_play = QPushButton("播放")
        self.btn_play.setCheckable(True) 
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        ctrl_layout.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.update_preview)
        ctrl_layout.addWidget(self.slider)
        
        self.frame_label = QLabel("0/0")
        self.frame_label.setFixedWidth(70)
        self.frame_label.setAlignment(Qt.AlignCenter)
        ctrl_layout.addWidget(self.frame_label)

        preview_layout.addLayout(ctrl_layout)
        preview_panel.setLayout(preview_layout)
        
        main_layout.addLayout(left_layout, 4)
        main_layout.addWidget(preview_panel, 5)
        self.setLayout(main_layout)

        self.data_source = None
        self.is_lazy_mode = False
        self.global_max = None

        self.update_ui_logic()

    def update_ui_logic(self):
        method = self.method_combo.currentText()
        current_codec = self.codec_combo.currentText()

        # 1. 级联更新编码器
        codecs_imageio = ['libx264', 'mpeg4']
        codecs_opencv = ['MJPG', 'XVID', 'HFYU']
        target_codecs = codecs_imageio if method == 'imageio' else codecs_opencv
        
        if self.codec_combo.count() != len(target_codecs) or self.codec_combo.itemText(0) != target_codecs[0]:
            self.codec_combo.blockSignals(True)
            self.codec_combo.clear()
            self.codec_combo.addItems(target_codecs)
            self.codec_combo.blockSignals(False)
            current_codec = target_codecs[0]

        # 2. 更新质量默认值
        if method == 'imageio':
            val = '18' if current_codec == 'libx264' else '3'
        else:
            val = '5' if current_codec != 'HFYU' else '0'
        self.qual_input.setText(val)

        # 3. 更新说明
        desc = "未知配置"
        if method == 'imageio':
            if current_codec == 'libx264':
                desc = "【libx264】H.264编码 (MP4)。兼容性最佳。\n质量值(CRF): 0(无损)~51(最差)，推荐18-23。"
            elif current_codec == 'mpeg4':
                desc = "【mpeg4】旧式编码 (AVI)。\n质量值(qscale): 1(最高)~31(最低)。"
        elif method == 'opencv':
            if current_codec == 'MJPG':
                desc = "【MJPG】Motion JPEG (AVI)。解码快但文件大。\n质量值: 1(最高)~31(最低)。"
            elif current_codec == 'XVID':
                desc = "【XVID】MPEG-4 Part 2 (AVI)。\n质量值: 1(最高)~31(最低)。"
            elif current_codec == 'HFYU':
                desc = "【HFYU】HuffYUV (AVI)。无损压缩，体积巨大。\n质量值: 无效。"
        
        self.info_label.setText(desc)

    def toggle_play(self):
        if self.btn_play.isChecked():
            self.btn_play.setText("暂停")
            try:
                fps = int(self.fps_input.text())
            except:
                fps = 30
            interval = max(1, int(1000 / fps))
            self.play_timer.start(interval)
        else:
            self.btn_play.setText("播放")
            self.play_timer.stop()

    def next_frame(self):
        if not self.slider.isEnabled():
            self.play_timer.stop()
            return
        
        curr = self.slider.value()
        max_val = self.slider.maximum()
        if curr < max_val:
            self.slider.setValue(curr + 1)
        else:
            self.slider.setValue(0)

    def update_preview(self, idx):
        if self.data_source is None:
            return

        total = len(self.data_source)
        self.frame_label.setText(f"{idx + 1}/{total}")
        
        if 0 <= idx < total:
            try:
                if self.is_lazy_mode:
                    frame = io.imread(self.data_source[idx])
                    img = normalize_single_frame(frame, global_max=None)
                else:
                    frame = self.data_source[idx]
                    img = normalize_single_frame(frame, global_max=self.global_max)

                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                elif img.ndim == 3 and img.shape[-1] == 1:
                    img = np.concatenate([img]*3, axis=-1)

                h, w, ch = img.shape
                bytes_per_line = ch * w
                qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                pix = QPixmap.fromImage(qimg).scaled(
                    self.img_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.img_preview.setPixmap(pix)
            except Exception as e:
                pass

    def load_data_ready(self, source, is_lazy, name_hint):
        self.data_source = source
        self.is_lazy_mode = is_lazy
        
        if not is_lazy and isinstance(source, np.ndarray):
            if source.dtype != np.uint8:
                self.global_max = source.max()
            else:
                self.global_max = 255
        else:
            self.global_max = None

        count = len(source)
        mode_str = "内存模式" if not is_lazy else "动态加载模式"
        self.input_label.setText(f"{name_hint} ({count} 帧) - {mode_str}")
        
        self.slider.setEnabled(True)
        self.slider.setRange(0, count - 1)
        self.slider.setValue(0)
        self.btn_play.setEnabled(True)
        self.update_preview(0)

    def select_img(self):
        """选择单文件(Stack)"""
        self.chk_lazy.setChecked(False) 
        
        # 打开时的默认路径：读取历史记录
        last_dir = self.settings.value("last_input_dir", ".")
        
        path, _ = QFileDialog.getOpenFileName(self, '选择 IMG Stack', last_dir, 'Images (*.tif *.tiff *.png *.jpg *.bmp)')
        
        if path:
            p_obj = Path(path)
            # 1. 更新历史记录
            self.settings.setValue("last_input_dir", str(p_obj.parent))
            # 2. 【关键】记录当前数据源目录，供保存时使用 (文件所在目录)
            self.current_source_dir = str(p_obj.parent)
            
            self.input_label.setText("正在读取...")
            QApplication.processEvents()
            try:
                stack = io.imread(path)
                if stack.ndim == 2: stack = stack[np.newaxis, ...]
                elif stack.ndim == 3 and stack.shape[-1] in [3, 4]: stack = stack[np.newaxis, ...]
                
                self.load_data_ready(stack, False, p_obj.name)
            except Exception as e:
                QMessageBox.critical(self, "读取失败", str(e))
                self.input_label.setText("读取失败")

    def select_folder(self):
        """选择文件夹(Sequence)"""
        last_dir = self.settings.value("last_input_dir", ".")
        
        folder = QFileDialog.getExistingDirectory(self, '选择序列文件夹', last_dir)
        
        if folder:
            p_obj = Path(folder)
            # 1. 更新历史记录
            self.settings.setValue("last_input_dir", str(p_obj.parent))
            # 2. 【关键】记录当前数据源目录 (文件夹的同级目录，即父目录)
            # 这样保存文件时，默认就会和这个 Sequence 文件夹在一起
            self.current_source_dir = str(p_obj.parent)
            
            self.input_label.setText("正在扫描...")
            QApplication.processEvents()
            
            files = natsorted([
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ])
            
            if not files:
                QMessageBox.warning(self, '警告', '无图像')
                return
            
            is_lazy = self.chk_lazy.isChecked()
            try:
                if is_lazy:
                    self.load_data_ready(files, True, p_obj.name)
                else:
                    stack = np.stack([io.imread(p) for p in files])
                    self.load_data_ready(stack, False, p_obj.name)
            except Exception as e:
                 QMessageBox.critical(self, "读取失败", f"错误: {e}")

    def select_output(self):
        """选择保存位置"""
        # 默认使用历史记录
        default_dir = self.settings.value("last_output_dir", ".")

        # 【关键】如果有当前加载的数据源目录，优先使用它
        # 逻辑：用户刚加载了 A 文件夹的图片，通常想把视频保存在 A 附近，而不是上次（可能是昨天）保存 B 项目的地方
        if self.current_source_dir and os.path.exists(self.current_source_dir):
            default_dir = self.current_source_dir
        
        folder = QFileDialog.getExistingDirectory(self, '选择保存文件夹', default_dir)
        
        if folder:
            # 保存这次的选择作为历史记录（备用）
            self.settings.setValue("last_output_dir", folder)
            
            self.out_folder = Path(folder)
            self.output_label.setText(self.out_folder.name)

    def start_conversion(self):
        if self.data_source is None or not hasattr(self, 'out_folder'):
            QMessageBox.warning(self, '提示', '请先加载数据并选择输出路径')
            return
        
        if self.btn_play.isChecked():
            self.toggle_play()

        try:
            fps = int(self.fps_input.text())
            quality = int(self.qual_input.text())
        except:
            QMessageBox.warning(self, '错误', 'FPS 和质量必须为整数')
            return

        filename = self.name_input.text().strip() or "output"
        ext = '.mp4' if self.method_combo.currentText() == 'imageio' else '.avi'
        out_path = self.out_folder / f"{filename}_{fps}fps{ext}"

        self.btn_convert.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = ImgToVideoWorker(
            self.data_source, out_path, fps,
            self.method_combo.currentText(),
            self.codec_combo.currentText(),
            quality,
            self.is_lazy_mode,
            self.global_max
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_finished(self, path):
        QMessageBox.information(self, '成功', f'视频已保存至:\n{path}')
        self.btn_convert.setEnabled(True)

    def on_error(self, msg):
        QMessageBox.critical(self, '失败', f'转换出错:\n{msg}')
        self.btn_convert.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win = ImgToVideoApp()
    win.show()
    sys.exit(app.exec_())