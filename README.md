# ImgToVideo (科学图像序列转视频工具)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**ImgToVideo** 是一个轻量级、无需安装的桌面应用程序，专为科研人员和工程师设计。它能够将医学图像 Stack（如 TIFF 多页文件）或图像序列文件夹（如 PNG/JPG 序列）快速转换为高质量的视频文件（MP4/AVI）。

该工具基于 Python (PyQt5) 开发，集成了 `imageio` (FFMPEG) 和 `OpenCV` 两种视频生成引擎，支持内存优化模式，可处理大规模数据集。

---

## ✨ 主要功能

* **多格式支持**：
  * 支持读取单文件 Stack：`.tif`, `.tiff` (多页 TIFF)。
  * 支持读取文件夹序列：`.png`, `.jpg`, `.bmp`, `.tif` 等。
* **双引擎内核**：
  * **ImageIO (FFMPEG)**：生成兼容性最好的 `.mp4` (H.264) 文件，支持 CRF 质量控制。
  * **OpenCV**：支持 `.avi` 格式，提供 MJPG、XVID 以及 HFYU（无损）编码。
* **内存优化 (Lazy Loading)**：
  * 支持“懒加载模式”，无需将所有图片一次性读入内存，适合处理数 GB 的大型图像序列。
* **实时预览**：
  * 内置播放器，支持滑动条拖动预览、暂停/播放、循环播放。
  * 自动归一化显示 16-bit 或高位深科学图像。
* **智能路径记忆**：
  * 自动记住上次打开的文件夹和保存位置，减少重复操作。

---

## 📥 下载与使用

1. pip install requirements.txt

2. 直接运行 img_to_video_gui.py文件

请访问右侧的 [Releases](../../releases) 页面下载最新版本的压缩包（如有上传）。

1. 下载 `.zip`。
2. 解压到任意文件夹。
3. 双击运行 `ImgToVideo.exe` 即可，无需安装 Python 环境。

---

## 🛠️ 开发环境搭建

如果你想自己修改代码或参与开发，请按以下步骤配置环境。

### 1. 克隆仓库

```bash
git clone [https://github.com/你的用户名/ImgToVideo.git](https://github.com/你的用户名/ImgToVideo.git)
cd ImgToVideo-Tool
```
