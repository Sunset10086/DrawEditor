import sys
import os
import math
import numpy as np
import random
import io
import zipfile
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QToolBar,
    QAction, QColorDialog, QFileDialog, QSpinBox, QSlider, QHBoxLayout,
    QVBoxLayout, QDialog, QDialogButtonBox, QFormLayout, QMessageBox,
    QScrollArea, QComboBox, QCheckBox, QLineEdit, QInputDialog, QListWidget,
    QListWidgetItem, QFontComboBox, QAbstractItemView, QDockWidget, QToolButton, QMenu,
    QGridLayout
)
from PyQt5.QtGui import (
    QImage, QPainter, QPen, QColor, QBrush, QCursor, QPixmap, QIcon,
    QPolygon, QFont, QFontMetrics, QPainterPath
)
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageFilter, ImageChops
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, QTimer, QEvent, QTranslator, QLocale, QLibraryInfo

def resource_path(rel):
    import sys, os
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, rel)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)

_grain_cache = {}
_grain_variants = 8
MASK_BRUSHES = ("round", "square", "airbrush", "grain", "grain_airbrush")

# 支持的图片扩展名
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".ico")
# 支持多帧的扩展名
MULTIFRAME_EXTS = (".gif", ".tif", ".tiff", ".webp")
# 项目文件扩展名
PROJECT_EXT = ".myp"

# ================= 图像转换 =================
def qimage_to_pil(qimg):
    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    
    return Image.frombytes("RGBA", (w, h), bytes(ptr))
def pil_to_qimage(pil_img):
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    data = pil_img.tobytes("raw", "RGBA")
    qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
    return qimg.copy()

def load_image_frames(path):
    """读取全部帧。返回 PIL Image 列表（RGBA）。若为单帧，返回长度为 1 的列表。"""
    frames = []
    img = Image.open(path)
    try:
        n = getattr(img, "n_frames", 1)
    except Exception:
        n = 1
    for i in range(n):
        try:
            img.seek(i)
        except EOFError:
            break
        frames.append(img.convert("RGBA").copy())
    if not frames:
        frames = [img.convert("RGBA").copy()]
    return frames

def save_project(path, stack, current_frame=0):
    """把图层栈保存为 zip：manifest.json + layerN.png"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        manifest = {
            "version": 1,
            "size": [stack.width(), stack.height()],
            "current": stack.current,
            "current_frame": current_frame,
            "layers": [],
        }
        for i, l in enumerate(stack.layers):
            name = f"layer{i}.png"
            buf = io.BytesIO()
            l.image.save(buf, "PNG")
            z.writestr(name, buf.getvalue())
            manifest["layers"].append({
                "name": l.name,
                "visible": bool(l.visible),
                "opacity": int(l.opacity),
                "blend": l.blend,
                "mask": None,      # 预留
                "file": name,
            })
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))


def load_project(path):
    """返回 (LayerStack, current_frame)"""
    with zipfile.ZipFile(path, "r") as z:
        manifest = json.loads(z.read("manifest.json").decode("utf-8"))
        w, h = manifest["size"]
        stack = LayerStack(w, h)
        stack.layers = []
        for info in manifest["layers"]:
            buf = io.BytesIO(z.read(info["file"]))
            img = Image.open(buf).convert("RGBA")
            layer = Layer(img, info["name"], info["visible"])
            layer.opacity = info.get("opacity", 100)
            layer.blend = info.get("blend", "normal")
            stack.layers.append(layer)
        stack.current = min(manifest.get("current", 0), len(stack.layers) - 1)
        return stack, manifest.get("current_frame", 0)

def load_image_first_frame(path):
    return load_image_frames(path)[0]

def make_checkerboard(w, h, size=8):
    """用 tile 平铺，避免逐像素 Python 循环。"""
    # 先造一个小 tile
    tile = QImage(size * 2, size * 2, QImage.Format_RGB32)
    c1 = QColor(200, 200, 200)
    c2 = QColor(255, 255, 255)
    for y in range(size * 2):
        for x in range(size * 2):
            color = c1 if ((x // size) + (y // size)) % 2 == 0 else c2
            tile.setPixelColor(x, y, color)
    # 平铺到 w×h
    pix = QPixmap(w, h)
    p = QPainter(pix)
    p.drawTiledPixmap(0, 0, w, h, QPixmap.fromImage(tile))
    p.end()
    return pix

# ================= 笔刷（带硬度） =================

_brush_cache = {}

def _get_grain_texture(size, variant=0):
    key = (size, variant)
    if key in _grain_cache:
        return _grain_cache[key]
    rng = np.random.default_rng(variant * 1000 + size)
    high = rng.random((size, size)).astype(np.float32)
    low_n = max(2, size // 4)
    low = rng.random((low_n, low_n)).astype(np.float32)
    from PIL import Image as _I
    low_img = _I.fromarray((low * 255).astype(np.uint8), "L").resize(
        (size, size), _I.BILINEAR)
    low_arr = np.asarray(low_img, dtype=np.float32) / 255.0
    tex = 0.6 * high + 0.4 * low_arr
    # 打薄：平方让低值更多，高值稀少
    tex = tex ** 2.2
    _grain_cache[key] = tex
    return tex

def get_brush_stamp(size, hardness, shape="round"):
    key = (size, hardness, shape)
    if key in _brush_cache:
        return _brush_cache[key]
    size = max(1, int(size))
    r = size / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2.0
    if shape == "square":
        dist = np.maximum(np.abs(xx - cx), np.abs(yy - cy))
    else:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    hard = max(0.0, min(1.0, hardness / 100.0))
    inner = r * hard
    if r - inner < 0.5:
        alpha = (dist <= r).astype(np.float32)
    else:
        alpha = np.clip((r - dist) / (r - inner), 0, 1).astype(np.float32)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 255
    rgba[..., 2] = 255
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    _brush_cache[key] = rgba
    return rgba

def stamp_brush(layer_img, x, y, color, size, hardness, erase=False,
                shape="round", grain=0.0, opacity=1.0, seed=0):
    size = max(1, int(size))
    stamp = get_brush_stamp(size, hardness, shape)
    x0 = int(x - size // 2)
    y0 = int(y - size // 2)
    lx0 = max(0, -x0); ly0 = max(0, -y0)
    lx1 = size - max(0, (x0 + size) - layer_img.width)
    ly1 = size - max(0, (y0 + size) - layer_img.height)
    if lx1 <= lx0 or ly1 <= ly0:
        return
    rx0 = max(0, x0); ry0 = max(0, y0)
    rx1 = rx0 + (lx1 - lx0); ry1 = ry0 + (ly1 - ly0)

    region = layer_img.crop((rx0, ry0, rx1, ry1))
    arr = np.array(region, dtype=np.uint8)
    sub = stamp[ly0:ly1, lx0:lx1]
    a = (sub[..., 3:4].astype(np.float32) / 255.0) * opacity

    # 颗粒调制：每次盖章用随机 variant
    if grain > 0:
        variant = (int(x) * 31 + int(y) * 17) % _grain_variants
        tex = _get_grain_texture(size, variant)[ly0:ly1, lx0:lx1]
        tex = tex[..., None]
        a = a * (1 - grain) + a * tex * grain * 2.0
        a = np.clip(a, 0, 1)

    if erase:
        arr[..., 3] = (arr[..., 3].astype(np.float32) * (1 - a[..., 0])).astype(np.uint8)
    else:
        src_rgb = np.array(color[:3], dtype=np.float32)
        src_a = (color[3] / 255.0) * a[..., 0]              # 0..1
        old_rgb = arr[..., :3].astype(np.float32)
        old_a = arr[..., 3:4].astype(np.float32) / 255.0    # 0..1

        sa = src_a[..., None]
        out_a = sa + old_a * (1.0 - sa)
        safe = np.where(out_a > 1e-6, out_a, 1.0)
        out_rgb = (src_rgb * sa + old_rgb * old_a * (1.0 - sa)) / safe

        arr[..., :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        arr[..., 3]  = np.clip(out_a[..., 0] * 255.0, 0, 255).astype(np.uint8)

    layer_img.paste(Image.fromarray(arr, "RGBA"), (rx0, ry0))

def draw_thick_line(layer_img, p1, p2, color, size, hardness,
                    erase=False, shape="round", grain=0.0,
                    opacity=1.0, seed=0):
    """沿线段密集盖章，各方向等粗"""
    x1, y1 = p1; x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    step = max(1.0, size * 0.1)
    n = max(1, int(dist / step))
    for i in range(n + 1):
        t = i / n
        stamp_brush(layer_img, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t,
                    color, size, hardness, erase, shape=shape, grain=grain, opacity=opacity, seed=seed)

def smudge_inplace(layer_img, x, y, size, strength, prev_x, prev_y):
    """原地涂抹：从 (prev_x, prev_y) 取源涂到 (x, y)。不返回新图，直接原地改。"""
    size = max(1, int(size))
    x0 = int(x - size // 2); y0 = int(y - size // 2)
    px0 = int(prev_x - size // 2); py0 = int(prev_y - size // 2)

    rx0 = max(0, x0); ry0 = max(0, y0)
    rx1 = min(layer_img.width, x0 + size); ry1 = min(layer_img.height, y0 + size)
    if rx1 <= rx0 or ry1 <= ry0:
        return

    # 目标区域在源中对应的位置
    src_x0 = px0 + (rx0 - x0)
    src_y0 = py0 + (ry0 - y0)
    src_x1 = src_x0 + (rx1 - rx0)
    src_y1 = src_y0 + (ry1 - ry0)
    if src_x0 < 0 or src_y0 < 0 or src_x1 > layer_img.width or src_y1 > layer_img.height:
        return

    src = layer_img.crop((src_x0, src_y0, src_x1, src_y1))
    dst = layer_img.crop((rx0, ry0, rx1, ry1))

    stamp = get_brush_stamp(size, 60)
    lx0 = rx0 - x0; ly0 = ry0 - y0
    sub = stamp[ly0:ly0 + (ry1 - ry0), lx0:lx0 + (rx1 - rx0), 3:4]\
        .astype(np.float32) / 255.0
    k = sub * strength

    s = np.asarray(src, dtype=np.float32)
    d = np.asarray(dst, dtype=np.float32)
    k = sub * strength                      # k 形状 (h, w, 1)
    s_a = s[..., 3:4] / 255.0               # 0..1
    d_a = d[..., 3:4] / 255.0
    w_s = s_a * k
    w_d = d_a * (1.0 - k)
    out_a = w_s + w_d                       # 0..1
    safe = np.where(out_a > 1e-6, out_a, 1.0)
    out_rgb = (s[..., :3] * w_s + d[..., :3] * w_d) / safe
    out = np.concatenate(
        [np.clip(out_rgb, 0, 255), np.clip(out_a * 255.0, 0, 255)], axis=-1)
    layer_img.paste(Image.fromarray(out.astype(np.uint8), "RGBA"), (rx0, ry0))

def draw_styled_line(layer_img, p1, p2, color, size, hardness, dash, arrow):
    """带虚线/箭头的直线"""
    x1, y1 = p1; x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    if dash == "solid":
        draw_thick_line(layer_img, p1, p2, color, size, hardness, False)
    else:
        on = size * 2 if dash == "dash" else size
        off = on
        t = 0.0
        drawing_on = True
        while t < length:
            step = on if drawing_on else off
            if drawing_on:
                a = t / length
                b = min((t + step) / length, 1.0)
                xa = x1 + (x2 - x1) * a; ya = y1 + (y2 - y1) * a
                xb = x1 + (x2 - x1) * b; yb = y1 + (y2 - y1) * b
                draw_thick_line(layer_img, (xa, ya), (xb, yb),
                                color, size, hardness, False)
            t += step
            drawing_on = not drawing_on
    # 箭头
    if arrow in ("end", "both"):
        _draw_arrow_tip(layer_img, (x1, y1), (x2, y2), color, size, hardness)
    if arrow == "both":
        _draw_arrow_tip(layer_img, (x2, y2), (x1, y1), color, size, hardness)

def _draw_arrow_tip(layer_img, from_pt, tip_pt, color, size, hardness):
    x1, y1 = from_pt
    x2, y2 = tip_pt
    angle = math.atan2(y2 - y1, x2 - x1)
    arr_size = max(8, size * 3)
    left = (x2 - arr_size * math.cos(angle - math.pi / 6),
            y2 - arr_size * math.sin(angle - math.pi / 6))
    right = (x2 - arr_size * math.cos(angle + math.pi / 6),
             y2 - arr_size * math.sin(angle + math.pi / 6))
    draw_thick_line(layer_img, (x2, y2), left, color, size, hardness, False)
    draw_thick_line(layer_img, (x2, y2), right, color, size, hardness, False)
    

# ================= 历史 =================

class History:
    def __init__(self, max_size=30):
        self.stack = []
        self.index = -1
        self.max_size = max_size

    def push(self, snapshot):
        self.stack = self.stack[:self.index + 1]
        self.stack.append(snapshot)
        if len(self.stack) > self.max_size:
            self.stack.pop(0)
        self.index = len(self.stack) - 1

    def undo(self):
        if self.index > 0:
            self.index -= 1
            return self.stack[self.index]
        return None

    def redo(self):
        if self.index < len(self.stack) - 1:
            self.index += 1
            return self.stack[self.index]
        return None

    def clear(self, snapshot):
        self.stack = [snapshot]
        self.index = 0

# ================= 图层 =================

class LayerStack:
    def __init__(self, w, h, bg=(255, 255, 255, 255)):
        self.layers = [Layer(Image.new("RGBA", (w, h), bg), "背景")]
        self.current = 0

    def width(self):
        return self.layers[0].image.width

    def height(self):
        return self.layers[0].image.height

    def cur(self):
        return self.layers[self.current]

    def snapshot(self):
        return [l.copy() for l in self.layers], self.current

    def restore(self, snap):
        layers, cur = snap
        self.layers = [l.copy() for l in layers]
        self.current = min(cur, len(self.layers) - 1)

    def flatten(self, extra_parts=None):
        w, h = self.width(), self.height()
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        for l in self.layers:
            if not l.visible:
                continue
            src = l.render()
            if l.blend == "normal" or out.getbbox() is None:
                out.alpha_composite(src)
            else:
                out = self._blend(out, src, l.blend)
        if extra_parts:
            for p in extra_parts:
                img = p.get("img")
                pos = p.get("pos")
                if img is None or pos is None:
                    continue
                out.alpha_composite(img, (int(pos.x()), int(pos.y())))
        return out

    def _blend(self, base, top, mode):
        """base、top 都是 RGBA，返回混合后的 RGBA"""
        b = np.array(base, dtype=np.float32) / 255.0
        t = np.array(top, dtype=np.float32) / 255.0
        # 取 top 的 alpha 作为混合权重
        ta = t[..., 3:4]
        ba = b[..., 3:4]
        out_a = ta + ba * (1 - ta)

        if mode == "multiply":
            blended = b[..., :3] * t[..., :3]
        elif mode == "screen":
            blended = 1 - (1 - b[..., :3]) * (1 - t[..., :3])
        elif mode == "overlay":
            blended = np.where(b[..., :3] < 0.5,
                               2 * b[..., :3] * t[..., :3],
                               1 - 2 * (1 - b[..., :3]) * (1 - t[..., :3]))
        else:
            blended = t[..., :3]

        # 用 top 的颜色作为混合结果的颜色
        out_rgb = (blended * ta + b[..., :3] * (1 - ta))
        # 归一化，避免除 0
        safe = np.where(out_a > 0, out_a, 1)
        out_rgb = out_rgb / safe
        out = np.concatenate([out_rgb, out_a], axis=-1)
        return Image.fromarray((out * 255).astype(np.uint8), "RGBA")

# ================= 浮动对象类型 =================
SEL_NONE = 0
SEL_MOVE = 1
SEL_SCALE = 2
SEL_ROTATE = 3

class SelectionGroup:
    """把多个子选区当做一个整体来操作；每个 part 持有自己的浮动图"""
    def __init__(self):
        # 每个 part: {"poly": QPolygon, "bbox": QRect,
        #            "img": PIL.Image, "pos": QPoint}
        self.parts = []
        self.group_bbox = None

    def is_empty(self):
        return not self.parts

    def recompute_bbox(self):
        if not self.parts:
            self.group_bbox = None
            return
        r = QRect(self.parts[0]["bbox"])
        for p in self.parts[1:]:
            r = r.united(p["bbox"])
        self.group_bbox = r

    def add_part(self, poly, bbox, img, pos):
        self.parts.append({
            "poly": QPolygon(poly),
            "bbox": QRect(bbox),
            "img": img,          # PIL.Image 或 None
            "pos": QPoint(pos) if pos is not None else None,
        })
        self.recompute_bbox()

    def contains(self, img_pt):
        for p in self.parts:
            if p["poly"].containsPoint(img_pt, Qt.OddEvenFill):
                return True
        return False

    def clear(self):
        self.parts = []
        self.group_bbox = None

# ================= 画布 =================

class Canvas(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.stack = LayerStack(800, 600)
        self.qimage = pil_to_qimage(self.stack.flatten())
        self._stroke_mask = None      # L 模式 Image，累积当前笔画的覆盖度（0-255）
        self._stroke_color = None     # 当前笔画颜色 (r, g, b, a)
        self._stroke_erase = False    # 当前笔画是否是擦除
        
        self.tool = "brush"
        self.drawing = False
        self.start_point = None
        self.last_point = None
        self.preview_shape = None
        self.current_btn = Qt.LeftButton

        # —— 选区整体 ——
        self.group = SelectionGroup()
        self.sel_mode = SEL_NONE
        self.sel_handle = None
        self.drag_start = None
        self.rotate_center = None
        self.rotate_start_angle = None
        self.original_parts = None
        self.original_float_pos = None

        self.float_meta = {}
        self.lasso_points = []
        self.zoom = 1.0
        self._checker = None
        self._pending_deselect = False

        # 线条编辑状态（直线/曲线共用）
        self.line_edit = False          # 是否处于线条编辑中
        self.line_p0 = None             # 起点（锚点）
        self.line_p3 = None             # 终点（锚点）
        self.line_ctrl1 = None          # 控制点1（直线模式下与 p0 重合）
        self.line_ctrl2 = None          # 控制点2（直线模式下与 p3 重合）
        self.line_drag = None           # 当前拖动目标：'p0' / 'p3' / 'c1' / 'c2' / None
        self.line_is_curve = False      # 是否曲线模式（用于控制点显示）
        self._line_preview_img = None

        self.brush_type = "round"       # 当前笔刷类型
        self.brush_opacity = 100        # 喷枪/整体不透明度
        self.grain_amount = 0.05         # 颗粒强度
        self.smudge_strength = 0.7      # 涂抹强度
        self._smudge_prev_pos = None

        self._air_timer = QTimer(self)
        self._air_timer.setInterval(150)          # 30ms ≈ 33Hz，越小越浓
        self._air_timer.timeout.connect(self._air_tick)
        self._air_hold_pos = None                # 鼠标当前按住的位置
        self._air_hold_color = None
        self._stroke_preview = None   # PIL Image，绘制过程中的临时层

        self.show_grid = False          # 是否显示网格
        self.grid_size = 16             # 网格间距（图像像素）
        self.grid_color = QColor(128, 128, 128, 90)   # 半透明灰
        self._panning = False
        self._pan_last = None
        self._stroke_preview_scaled = None
        self._stroke_preview_zoom = None

    def _rebuild_line_preview(self):
        if not self.line_edit or self.line_p0 is None or self.line_p3 is None:
            self._line_preview_img = None
            return
        w, h = self.stack.width(), self.stack.height()
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        if self.line_is_curve:
            samples = 80
            p0, p1, p2, p3 = (self.line_p0, self.line_ctrl1,
                              self.line_ctrl2, self.line_p3)
            pts = []
            for i in range(samples + 1):
                t = i / samples
                x = ((1-t)**3 * p0.x() + 3*(1-t)**2 * t * p1.x()
                     + 3*(1-t) * t**2 * p2.x() + t**3 * p3.x())
                y = ((1-t)**3 * p0.y() + 3*(1-t)**2 * t * p1.y()
                     + 3*(1-t) * t**2 * p2.y() + t**3 * p3.y())
                pts.append((int(x), int(y)))
        else:
            pts = [(self.line_p0.x(), self.line_p0.y()),
                   (self.line_p3.x(), self.line_p3.y())]

        color = self.mw.foreground
        dash = self.mw.line_dash
        arrow = self.mw.line_arrow

        for (xa, ya, xb, yb) in self._styled_segments(pts, dash):
            draw_thick_line(img, (xa, ya), (xb, yb),
                            color.getRgb(), self.mw.brush_size,
                            self.mw.brush_hardness, False)
        if arrow in ("end", "both") and len(pts) >= 2:
            _draw_arrow_tip(img, pts[-2], pts[-1], color.getRgb(),
                            self.mw.brush_size, self.mw.brush_hardness)
        if arrow == "both" and len(pts) >= 2:
            _draw_arrow_tip(img, pts[1], pts[0], color.getRgb(),
                            self.mw.brush_size, self.mw.brush_hardness)

        self._line_preview_img = img

    def ensure_canvas_size(self, w, h):
        """确保画布至少 w×h；不够就扩展每个图层，保留原内容居中"""
        cur_w = self.stack.width()
        cur_h = self.stack.height()
        if cur_w >= w and cur_h >= h:
            return
        new_w = max(cur_w, w)
        new_h = max(cur_h, h)
        # 原有内容向左上角对齐（跟粘贴的坐标计算一致）
        for l in self.stack.layers:
            new_img = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
            new_img.alpha_composite(l.image, (0, 0))
            l.image = new_img
        # 画布的浮动 part 无需改动（它们用绝对坐标）
        self.resize(self.sizeHint())
        self.refresh()
        
    # ---------- 尺寸 / 刷新 ----------     
    def _air_tick(self):
        if self._air_hold_pos is None or self._air_hold_color is None:
            return
        self._grain_air_batch([(self._air_hold_pos.x(),
                                self._air_hold_pos.y(),
                                self._air_hold_color)])
        self.update()
        
    def _grain_air_batch(self, samples):
        if not samples:
            return
        size = max(1, int(self.mw.brush_size))
        h = self.mw.brush_hardness / 100.0
        scatter = max(1.0, size * 0.35)
        opacity = self.brush_opacity / 100.0 * 0.6
        grain_total = min(1.0, self.grain_amount + h)

        pts = []
        for (sx, sy, color) in samples:
            col = color.getRgb() if color is not None else (0, 0, 0, 255)
            for _ in range(int(size * 0.25)):
                dx = random.uniform(-scatter, scatter)
                dy = random.uniform(-scatter, scatter)
                pts.append((int(sx + dx), int(sy + dy), col))
        if not pts:
            return

        if self._stroke_mask is not None:
            # 无论有无选区，都累积到 _stroke_mask
            w, h = self.stack.width(), self.stack.height()
            temp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            self._grain_air_draw(temp, pts, size, opacity, grain_total)
            seg_a = temp.split()[3]
            self._stroke_mask = ImageChops.lighter(self._stroke_mask, seg_a)
            return

        # 有选区或没 mask：原逻辑
        if self.group.is_empty():
            self._grain_air_draw(
                self.stack.cur().image, pts, size, opacity, grain_total)
            return
        for p in self.group.parts:
            img = p.get("img")
            pos = p.get("pos")
            if img is None or pos is None:
                continue
            local_pts = [(px - pos.x(), py - pos.y(), col)
                         for (px, py, col) in pts]
            self._grain_air_draw(img, local_pts, size, opacity, grain_total)

    def _grain_air_draw(self, target_img, pts, size, opacity, grain_total):
        """把 pts 批量合成到 target_img 上（pts 为 target 的局部坐标）"""
        if not pts:
            return
        pad = size // 2 + 2
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x = max(0, min(xs) - pad)
        min_y = max(0, min(ys) - pad)
        max_x = min(target_img.width, max(xs) + pad)
        max_y = min(target_img.height, max(ys) + pad)
        bw = max_x - min_x
        bh = max_y - min_y
        if bw <= 0 or bh <= 0:
            return

        # 所有颗粒先撒到一张小临时图上
        temp = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        grain_size = min(int(size/3), int((100 - self.mw.brush_hardness)/5))
        for (px, py, col) in pts:
            stamp_brush(temp, px - min_x, py - min_y, col,
                        grain_size, 100, False,
                        shape="round", grain=grain_total, opacity=opacity)

        # 一次性叠回 target
        region = target_img.crop((min_x, min_y, max_x, max_y))
        region.alpha_composite(temp)
        target_img.paste(region, (min_x, min_y))
        
    def sizeHint(self):
        return QSize(int(self.stack.width() * self.zoom),
                     int(self.stack.height() * self.zoom))

    def minimumSizeHint(self):
        return self.sizeHint()

    def refresh_image(self):
        if (self._stroke_mask is not None and self._stroke_erase):
            mask_eff = self._effective_stroke_mask()
            inv = mask_eff.point(lambda v: 255 - v)
            layer = self.stack.cur()
            saved = layer.image
            r, g, b, a = saved.split()
            a2 = ImageChops.multiply(a, inv)
            layer.image = Image.merge("RGBA", (r, g, b, a2))
            try:
                self.qimage = pil_to_qimage(self.stack.flatten())
            finally:
                layer.image = saved
        else:
            self.qimage = pil_to_qimage(self.stack.flatten())

    def refresh(self):
        self.refresh_image()
        self.resize(self.sizeHint())
        self.update()

    def refresh_view(self):
        """缩放/视口变化：只重建缩放缓存 + resize + update"""
        self._stroke_preview_scaled = None
        self.resize(self.sizeHint())
        self.update()

    def set_image(self, pil_img, reset_history=True):
        self.commit_all()
        w, h = pil_img.width, pil_img.height
        self.stack = LayerStack(w, h)
        self.stack.layers[0].image = pil_img.convert("RGBA")
        self.group.clear()
        if reset_history:
            self.mw.history.clear(self.snapshot())
        self.resize(self.sizeHint())
        self.refresh()

    def snapshot(self):
        layers_snap = self.stack.snapshot()
        group_snap = [{
            "poly": QPolygon(p["poly"]),
            "bbox": QRect(p["bbox"]),
            "img": p["img"].copy() if p.get("img") is not None else None,
            "pos": QPoint(p["pos"]) if p.get("pos") is not None else None,
        } for p in self.group.parts]
        # 帧状态
        mw = self.mw
        frames_snap = None
        if getattr(mw, "_multiframe", False):
            frames_snap = [f.copy() for f in mw._frames]
        return {
            "layers": layers_snap,
            "group": group_snap,
            "multiframe": getattr(mw, "_multiframe", False),
            "frames": frames_snap,
            "frame_index": getattr(mw, "_frame_index", 0),
        }

    def restore(self, snap):
        layers_snap = snap["layers"]
        group_snap = snap["group"]
        self.stack.restore(layers_snap)
        self.group.parts = group_snap
        self.group.recompute_bbox()
        # 恢复帧状态
        mw = self.mw
        is_mf = snap.get("multiframe", False)
        frames = snap.get("frames")
        # 先处理"进入/退出多帧"的状态切换
        if is_mf != getattr(mw, "_multiframe", False):
            if is_mf:
                # 恢复为多帧：把 frames 塞回去
                mw._multiframe = True
                mw._frames = [f.copy() for f in (frames or [])]
                mw._frame_index = snap.get("frame_index", 0)
                mw.layer_panel.set_multiframe(True, mw)
                mw._set_mode_menu_titles("帧")
            else:
                mw._multiframe = False
                mw._frames = []
                mw._frame_index = 0
                mw.layer_panel.set_multiframe(False, mw)
                mw._set_mode_menu_titles("图层")
        elif is_mf:
            # 仍然在多帧模式：恢复 frames 列表本身
            mw._frames = [f.copy() for f in (frames or [])]
            mw._frame_index = snap.get("frame_index", 0)
        # 刷新面板
        if is_mf:
            mw._update_frame_list()
        else:
            mw.layer_panel.stack = self.stack
            mw.layer_panel.refresh_list()
        self.refresh()

    def _notify_thumb(self):
        if not hasattr(self, "_thumb_timer"):
            from PyQt5.QtCore import QTimer
            self._thumb_timer = QTimer(self)
            self._thumb_timer.setSingleShot(True)
            self._thumb_timer.setInterval(80)
            self._thumb_timer.timeout.connect(
                lambda: self.mw.layer_panel.refresh_current_thumb())
        self._thumb_timer.start()

    # ---------- 坐标 ----------
    def widget_to_image(self, pos):
        return QPoint(int(pos.x() / self.zoom), int(pos.y() / self.zoom))

    def image_to_widget(self, pt):
        return QPoint(int(pt.x() * self.zoom), int(pt.y() * self.zoom))

    # ---------- 绘制 ----------
    def _make_rotate_cursor(self):
        d = 24
        pm = QPixmap(d, d)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(0, 0, 0), 2))
        # 画一个圆弧 + 箭头
        from PyQt5.QtCore import QRectF
        p.drawArc(QRectF(4, 4, 16, 16), 30 * 16, 300 * 16)
        # 箭头
        p.drawLine(16, 2, 19, 5)
        p.drawLine(16, 2, 13, 5)
        p.end()
        return QCursor(pm, d // 2, d // 2)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        w = int(self.qimage.width() * self.zoom)
        h = int(self.qimage.height() * self.zoom)

        # 棋盘底
        if (self._checker is None or self._checker.width() != w
                or self._checker.height() != h):
            self._checker = make_checkerboard(w, h, 8)
        painter.drawPixmap(0, 0, self._checker)

        # 主图
        painter.drawImage(QRect(0, 0, w, h), self.qimage)

        # 像素网格
        if self.show_grid and self.grid_size > 0:
            step_px = self.grid_size * self.zoom
            if step_px >= 3:
                pen = QPen(self.grid_color)
                pen.setWidth(1)
                painter.setPen(pen)
                x = 0.0
                while x <= w:
                    painter.drawLine(int(x), 0, int(x), h)
                    x += step_px
                y = 0.0
                while y <= h:
                    painter.drawLine(0, int(y), w, int(y))
                    y += step_px

        # ---------- 浮动 part ----------
        # 擦除中：整图 mask（按 opacity 缩放），用于实时削减浮动图 alpha
        erase_mask_full = None
        if (self._stroke_mask is not None and self._stroke_erase
                and not self.group.is_empty()):
            erase_mask_full = self._effective_stroke_mask()

        for part in self.group.parts:
            img = part.get("img")
            pos = part.get("pos")
            if img is None or pos is None:
                continue
            disp = img
            if erase_mask_full is not None:
                x0, y0 = pos.x(), pos.y()
                x1, y1 = x0 + img.width, y0 + img.height
                # 局部 mask，只保留该 part 多边形内
                local_mask = Image.new("L", (img.width, img.height), 0)
                ImageDraw.Draw(local_mask).polygon(
                    [(q.x() - x0, q.y() - y0) for q in part["poly"]], fill=255)
                sub = erase_mask_full.crop((x0, y0, x1, y1))
                clipped = ImageChops.multiply(sub, local_mask)
                inv = clipped.point(lambda v: 255 - v)
                r, g, b, a = img.split()
                a2 = ImageChops.multiply(a, inv)
                disp = Image.merge("RGBA", (r, g, b, a2))
            fq = pil_to_qimage(disp)
            frect = QRect(
                int(pos.x() * self.zoom),
                int(pos.y() * self.zoom),
                max(1, int(fq.width() * self.zoom)),
                max(1, int(fq.height() * self.zoom)))
            painter.drawImage(frect, fq)

        # ---------- 绘制中笔画缓存 ----------
        if self._stroke_preview is not None:
            sp_q = pil_to_qimage(self._stroke_preview)
            painter.drawImage(QRect(0, 0, w, h), sp_q)

        # ---------- 笔画 mask 预览（非擦除） ----------
        if self._stroke_mask is not None and not self._stroke_erase:
            rgba = Image.new("RGBA", self._stroke_mask.size, self._stroke_color)
            rgba.putalpha(self._stroke_mask)
            if self.brush_opacity < 100:
                a = rgba.split()[3].point(
                    lambda v: int(v * self.brush_opacity / 100))
                rgba.putalpha(a)
            sp_q = pil_to_qimage(rgba)
            painter.drawImage(QRect(0, 0, w, h), sp_q)

        # ---------- preview_shape ----------
        if self.preview_shape is not None:
            kind = self.preview_shape[0]

            if kind == "shape":
                _, r = self.preview_shape
                pen = QPen(self.mw.foreground,
                           max(1, int(self.mw.brush_size * self.zoom)))
                painter.setPen(pen)
                painter.setBrush(QBrush(self.mw.background)
                                 if self.mw.shape_fill else Qt.NoBrush)
                pad = self.mw.brush_size + 2
                w2 = max(1, r.width())
                h2 = max(1, r.height())
                pts = self._shape_polygon(self.mw.shape_kind, w2, h2, pad)
                wpoly = QPolygon()
                for (px, py) in pts:
                    ix = r.left() + (px - pad)
                    iy = r.top() + (py - pad)
                    wpoly.append(self.image_to_widget(QPoint(int(ix), int(iy))))
                painter.drawPolygon(wpoly)

            elif kind == "sel_rect":
                _, r = self.preview_shape
                pen = QPen(QColor(0, 0, 0), 1, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRect(self.image_to_widget(r.topLeft()),
                                       self.image_to_widget(r.bottomRight())))

            elif kind == "sel_ellipse":
                _, r = self.preview_shape
                pen = QPen(QColor(0, 0, 0), 1, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                wr = QRect(self.image_to_widget(r.topLeft()),
                           self.image_to_widget(r.bottomRight()))
                painter.drawEllipse(wr)

            elif kind == "sel_polyline":
                _, pts = self.preview_shape
                pen = QPen(QColor(0, 0, 0), 1, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                for i in range(len(pts) - 1):
                    painter.drawLine(self.image_to_widget(pts[i]),
                                     self.image_to_widget(pts[i + 1]))
                if len(pts) >= 3:
                    painter.drawLine(self.image_to_widget(pts[-1]),
                                     self.image_to_widget(pts[0]))

            elif kind == "grad_line":
                _, p1, p2 = self.preview_shape
                painter.setPen(QPen(QColor(0, 0, 0), 1, Qt.DashLine))
                painter.drawLine(self.image_to_widget(p1),
                                 self.image_to_widget(p2))

            elif kind == "line_edit":
                # 1) 线条预览图（PIL，画布大小）——与最终结果一致
                if getattr(self, "_line_preview_img", None) is not None:
                    q = pil_to_qimage(self._line_preview_img)
                    painter.drawImage(QRect(0, 0, w, h), q)

                # 2) 控制柄（覆盖在预览之上）
                if self.line_p0 is not None and self.line_p3 is not None:
                    if self.line_is_curve:
                        hp = QPen(QColor(0, 120, 215), 1, Qt.DashLine)
                        painter.setPen(hp)
                        painter.setBrush(Qt.NoBrush)
                        painter.drawLine(self.image_to_widget(self.line_p0),
                                         self.image_to_widget(self.line_ctrl1))
                        painter.drawLine(self.image_to_widget(self.line_p3),
                                         self.image_to_widget(self.line_ctrl2))

                    painter.setBrush(QBrush(QColor(0, 120, 215)))
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    for pt in (self.line_p0, self.line_p3):
                        w2 = self.image_to_widget(pt)
                        painter.drawEllipse(QRect(w2.x() - 5, w2.y() - 5, 10, 10))

                    if self.line_is_curve:
                        painter.setBrush(QBrush(QColor(255, 200, 0)))
                        painter.setPen(QPen(QColor(255, 255, 255), 1))
                        for pt in (self.line_ctrl1, self.line_ctrl2):
                            w2 = self.image_to_widget(pt)
                            painter.drawRect(QRect(w2.x() - 5, w2.y() - 5, 10, 10))

        # ---------- 子选区虚线 + 整体框 ----------
        if not self.group.is_empty():
            pen = QPen(QColor(0, 0, 0), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            for p in self.group.parts:
                wpoly = QPolygon([self.image_to_widget(q) for q in p["poly"]])
                painter.drawPolygon(wpoly)

            if self.group.group_bbox is not None:
                self._draw_handles(painter, self.group.group_bbox)

        painter.end()
            
        # 子选区虚线
        if not self.group.is_empty():
            pen = QPen(QColor(0, 0, 0), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            for p in self.group.parts:
                wpoly = QPolygon([self.image_to_widget(q) for q in p["poly"]])
                painter.drawPolygon(wpoly)

            if self.group.group_bbox is not None:
                self._draw_handles(painter, self.group.group_bbox)

        painter.end()

    def _preview_arrow(self, painter, from_pt, tip_pt):
        x1, y1 = from_pt
        x2, y2 = tip_pt
        angle = math.atan2(y2 - y1, x2 - x1)
        size = max(8, self.mw.brush_size * 3)
        left = (x2 - size * math.cos(angle - math.pi / 6),
                y2 - size * math.sin(angle - math.pi / 6))
        right = (x2 - size * math.cos(angle + math.pi / 6),
                 y2 - size * math.sin(angle + math.pi / 6))
        painter.drawLine(self.image_to_widget(QPoint(int(x2), int(y2))),
                         self.image_to_widget(QPoint(int(left[0]), int(left[1]))))
        painter.drawLine(self.image_to_widget(QPoint(int(x2), int(y2))),
                         self.image_to_widget(QPoint(int(right[0]), int(right[1]))))

    def _draw_handles(self, painter, bbox):
        handles = self._handles_for(bbox)
        rot = self._rotate_for(bbox)
        painter.setBrush(QBrush(QColor(0, 120, 215)))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        for h in handles:
            painter.drawRect(QRect(h.x() - 4, h.y() - 4, 8, 8))
        painter.drawEllipse(QRect(rot.x() - 5, rot.y() - 5, 10, 10))

    def _handles_for(self, bbox):
        r = QRect(self.image_to_widget(bbox.topLeft()),
                  self.image_to_widget(bbox.bottomRight()))
        return [
            QPoint(r.left(), r.top()),
            QPoint(r.center().x(), r.top()),
            QPoint(r.right(), r.top()),
            QPoint(r.left(), r.center().y()),
            QPoint(r.right(), r.center().y()),
            QPoint(r.left(), r.bottom()),
            QPoint(r.center().x(), r.bottom()),
            QPoint(r.right(), r.bottom()),
        ]

    def _rotate_for(self, bbox):
        r = QRect(self.image_to_widget(bbox.topLeft()),
                  self.image_to_widget(bbox.bottomRight()))
        return QPoint(r.center().x(), r.top() - 24)

    # ---------- 光标 ----------
    def _make_circle_cursor(self, diameter):
        d = max(4, int(diameter))
        pm = QPixmap(d + 2, d + 2)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.drawEllipse(1, 1, d, d)
        p.setPen(QPen(QColor(255, 255, 255), 1, Qt.DotLine))
        p.drawEllipse(1, 1, d, d)
        p.end()
        return QCursor(pm, d // 2 + 1, d // 2 + 1)

    def _make_brush_cursor(self):
        if self.brush_type == "square":
            d = max(4, int(self.mw.brush_size * self.zoom))
            pm = QPixmap(d + 2, d + 2)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.drawRect(1, 1, d, d)
            p.end()
            return QCursor(pm, d // 2 + 1, d // 2 + 1)
        d = max(4, int(self.mw.brush_size * self.zoom))
        return self._make_circle_cursor(d)

    def _update_cursor(self, pos):
        # 1) 手柄 / 旋转柄命中：无论什么工具都优先显示缩放/旋转光标
        if not self.group.is_empty():
            hit = self._hit_test_group(pos)
            if hit == "rotate":
                self.setCursor(self._make_rotate_cursor()); return
            if hit in (0, 7): self.setCursor(Qt.SizeFDiagCursor); return
            if hit in (2, 5): self.setCursor(Qt.SizeBDiagCursor); return
            if hit in (1, 6): self.setCursor(Qt.SizeVerCursor); return
            if hit in (3, 4): self.setCursor(Qt.SizeHorCursor); return

        # 2) 选区工具：选区内显示"移动"光标，选区外显示"十字"
        if not self.group.is_empty() and self.tool in ("ellipse_select", "rect_select", "lasso",
                                                       "shape", "text"):
            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                self.setCursor(Qt.CrossCursor); return
            img_pt = self.widget_to_image(pos)
            if self.group.contains(img_pt):
                self.setCursor(Qt.SizeAllCursor); return
            self.setCursor(Qt.CrossCursor); return

        # 3) 线条工具编辑中：命中端点/控制点显示"移动"，否则"十字"
        if self.tool == "line" and self.line_edit:
            hit = self._hit_test_line_handle(pos)
            if hit in ("p0", "p3", "c1", "c2"):
                self.setCursor(Qt.SizeAllCursor); return
            self.setCursor(Qt.CrossCursor); return
            
        if self.tool == "pan":
            self.setCursor(Qt.OpenHandCursor)
            return

        # 4) 其他工具走正常光标
        if self.tool in ("brush", "eraser"):
            self.setCursor(self._make_brush_cursor())
        elif self.tool in ("picker", "fill"):
            self.setCursor(Qt.CrossCursor)
        elif self.tool == "text":
            self.setCursor(Qt.IBeamCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    # ---------- 鼠标按下 ----------
    def mousePressEvent(self, event):
        pos = event.pos()
        img_pt = self.widget_to_image(pos)
        btn = event.button()
        if btn not in (Qt.LeftButton, Qt.RightButton):
            return
        use_bg = (btn == Qt.RightButton)
        self.current_btn = btn

        if self.tool == "pan":
            self._panning = True
            self._pan_last = pos
            self.setCursor(Qt.ClosedHandCursor)
            return

        # —— 选取工具：操作已有选区 ——
        if not self.group.is_empty():
            ctrl = bool(event.modifiers() & Qt.ControlModifier)
            hit = self._hit_test_group(pos)
            bbox = self.group.group_bbox
            if hit == "rotate":
                self.sel_mode = SEL_ROTATE
                self.drag_start = pos
                self.rotate_center = bbox.center()
                self.rotate_start_angle = math.atan2(
                    pos.y() - self.rotate_center.y(),
                    pos.x() - self.rotate_center.x())
                self._save_original_group()
                return
            if hit is not None:
                self.sel_mode = SEL_SCALE
                self.sel_handle = hit
                self.drag_start = pos
                self._save_original_group()
                return
            # ↓↓↓ 关键修改：只有选区工具才"点内部=移动选区"
            if (not ctrl
                    and self.tool in ("rect_select", "ellipse_select", "lasso", "shape", "text")
                    and (self.group.contains(img_pt) or bbox.contains(img_pt))):
                self.sel_mode = SEL_MOVE
                self.drag_start = pos
                self._save_original_group()
                return
            if not ctrl and self.tool in ("rect_select", "lasso", "ellipse_select"):
                self._pending_deselect = True

        # —— 非选取/非形状工具 + 已有选区：只允许选区内操作 ——
        if (self.tool not in ("ellipse_select", "rect_select", "lasso", "shape")
                and not self.group.is_empty()):
            in_sel = self.group.contains(img_pt) or (
                self.group.group_bbox and self.group.group_bbox.contains(img_pt))
            if not in_sel:
                self.drawing = False
                return

        # —— 正式绘制 ——
        self.start_point = img_pt
        self.last_point = img_pt
        self.drawing = True

        if self.tool in ("brush", "eraser"):
            self._smudge_prev_pos = None
            if self.tool == "brush" and self.brush_type == "grain_airbrush":
                self._air_hold_pos = img_pt
                self._air_hold_color = (
                    self.mw.background if use_bg else self.mw.foreground)
                self._air_timer.start()
            color = self.mw.background if use_bg else self.mw.foreground

            if self.brush_type in MASK_BRUSHES:
                w, h = self.stack.width(), self.stack.height()
                self._stroke_mask = Image.new("L", (w, h), 0)
                self._stroke_color = (0, 0, 0, 255) if color is None else color.getRgb()
                self._stroke_erase = (self.tool == "eraser")
            else:
                if self.group.is_empty() and self.brush_type != "smudge":
                    w, h = self.stack.width(), self.stack.height()
                    self._stroke_preview = Image.new("RGBA", (w, h), (0, 0, 0, 0))

            if self.tool == "eraser":
                self._stroke(img_pt, img_pt, None, erase=True)
            else:
                self._stroke(img_pt, img_pt, color, erase=False)
            self.update()
    
        elif self.tool == "line":
            # 已有编辑对象：判断是否命中端点或控制点
            if self.line_edit:
                hit = self._hit_test_line_handle(pos)
                if hit:
                    self.line_drag = hit
                    self.drawing = True
                    return
                # 点在所有控制柄之外：提交
                self._commit_line()
                return
            # 开始新线条
            self.line_p0 = QPoint(img_pt)
            self.line_p3 = QPoint(img_pt)
            self.line_ctrl1 = QPoint(img_pt)
            self.line_ctrl2 = QPoint(img_pt)
            self.line_is_curve = (self.mw.line_style == "curve")
            self.line_edit = True
            self.preview_shape = ("line_edit",)
            self._rebuild_line_preview()
            self.update()
        elif self.tool in ("rect", "ellipse"):
            self.preview_shape = (self.tool, QRect(img_pt, img_pt))
            self.update()
        elif self.tool == "picker":
            self.pick_color(pos, secondary=use_bg)
            self.drawing = False
        elif self.tool == "fill":
            color = self.mw.background if use_bg else self.mw.foreground
            def do(img):
                self.flood_fill(img, img_pt, color)
                return img
            self._apply_to_layer_in_selection(do)
            self.mw.push_history()
            self.refresh()
            self.drawing = False
        elif self.tool == "ellipse_select":
            self.start_point = img_pt
        elif self.tool == "rect_select":
            self.start_point = img_pt
        elif self.tool == "lasso":
            self.start_point = img_pt
            self.lasso_points = [img_pt]
        elif self.tool == "text":
            self.drawing = False
            self._insert_text(img_pt)
        elif self.tool == "gradient":
            self.start_point = img_pt
            self.preview_shape = ("grad_line", img_pt, img_pt)
            self.update()
        elif self.tool == "shape":
            self.preview_shape = ("shape", QRect(img_pt, img_pt))
            self.update()

    def _save_original_group(self):
        self.original_parts = [
            {"poly": QPolygon(p["poly"]),
             "bbox": QRect(p["bbox"]),
             "img": p.get("img").copy() if p.get("img") is not None else None,
             "pos": QPoint(p["pos"]) if p.get("pos") is not None else None}
            for p in self.group.parts]

    def _hit_test_group(self, widget_pos):
        bbox = self.group.group_bbox
        if bbox is None:
            return None
        rot = self._rotate_for(bbox)
        if (widget_pos - rot).manhattanLength() < 12:
            return "rotate"
        for i, h in enumerate(self._handles_for(bbox)):
            if (widget_pos - h).manhattanLength() < 12:
                return i
        return None

    # ---------- 鼠标移动 ----------
    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self.tool == "pan" and self._panning and self._pan_last is not None:
            delta = pos - self._pan_last
            self._pan_last = pos
            sa = self.mw.scroll
            sa.horizontalScrollBar().setValue(
                sa.horizontalScrollBar().value() - delta.x())
            sa.verticalScrollBar().setValue(
                sa.verticalScrollBar().value() - delta.y())
            return
        
        img_pt = self.widget_to_image(pos)
        self.mw.update_pos_label(img_pt, self.stack.flatten())

        if self.sel_mode == SEL_MOVE and self.original_parts is not None:
            delta = pos - self.drag_start
            di = QPoint(int(delta.x() / self.zoom), int(delta.y() / self.zoom))
            for i, p in enumerate(self.group.parts):
                p["poly"] = QPolygon([q + di for q in self.original_parts[i]["poly"]])
                p["bbox"] = p["poly"].boundingRect()
                op = self.original_parts[i].get("pos")
                p["pos"] = (op + di) if op is not None else None
            self.group.recompute_bbox()
            self.update(); return

        if self.sel_mode == SEL_SCALE:
            self._apply_scale_group(pos, event.modifiers()); self.update(); return
        if self.sel_mode == SEL_ROTATE:
            self._apply_rotate_group(pos); self.update(); return

        if not self.drawing:
            self._update_cursor(pos)
            return

        if self.tool in ("brush", "eraser"):
            if self.tool == "eraser":
                self._stroke(self.last_point, img_pt, None, erase=True)
            else:
                color = self.mw.background if self.current_btn == Qt.RightButton \
                    else self.mw.foreground
                self._stroke(self.last_point, img_pt, color, erase=False)
            self.last_point = img_pt
            if self.tool == "brush" and self.brush_type == "grain_airbrush":
                self._air_hold_pos = img_pt
            if self.tool == "eraser":
                self.refresh()          # 擦除：直接改图层，需要 refresh
            else:
                self.update()                # 只 update，不 refresh
        elif self.tool == "line":
            if not self.line_edit:
                return
            if self.line_drag == "p0":
                self.line_p0 = QPoint(img_pt)
            elif self.line_drag == "p3":
                self.line_p3 = QPoint(img_pt)
            elif self.line_drag == "c1":
                self.line_ctrl1 = QPoint(img_pt)
            elif self.line_drag == "c2":
                self.line_ctrl2 = QPoint(img_pt)
            elif self.line_drag is None and self.start_point is not None:
                # 还在拖基线：更新终点
                self.line_p3 = QPoint(img_pt)
                # 曲线模式下默认控制点放在 1/3 和 2/3；直线模式下控制点跟随端点
                if self.line_is_curve:
                    self.line_ctrl1 = QPoint(
                        self.line_p0.x() + (img_pt.x() - self.line_p0.x()) // 3,
                        self.line_p0.y() + (img_pt.y() - self.line_p0.y()) // 3)
                    self.line_ctrl2 = QPoint(
                        self.line_p0.x() + 2 * (img_pt.x() - self.line_p0.x()) // 3,
                        self.line_p0.y() + 2 * (img_pt.y() - self.line_p0.y()) // 3)
                else:
                    self.line_ctrl1 = QPoint(self.line_p0)
                    self.line_ctrl2 = QPoint(img_pt)
            self._rebuild_line_preview()
            self.update()
        elif self.tool in ("rect", "ellipse"):
            rect = self._maybe_square(self.start_point, img_pt, event.modifiers())
            self.preview_shape = (self.tool, rect); self.update()
        elif self.tool == "ellipse_select":
            if self.drawing and self.start_point is not None:
                rect = self._maybe_square(self.start_point, img_pt,event.modifiers())
                self.preview_shape = ("sel_ellipse", rect)
                self.update()
        elif self.tool == "rect_select":
            if self.drawing and self.start_point is not None:
                rect = self._maybe_square(self.start_point, img_pt,
                                          event.modifiers())
                self.preview_shape = ("sel_rect", rect)
                self.update()
        elif self.tool == "lasso":
            if self.drawing:
                if not self.lasso_points:
                    self.lasso_points = [self.start_point]
                last = self.lasso_points[-1]
                # 距离阈值，避免采样过密
                if (img_pt - last).manhattanLength() >= 2:
                    self.lasso_points.append(img_pt)
                # 预览把当前鼠标点也加进去
                preview_pts = list(self.lasso_points) + [img_pt]
                self.preview_shape = ("sel_polyline", preview_pts)
                self.update()
        elif self.tool == "gradient":
            self.preview_shape = ("grad_line", self.start_point, img_pt)
            self.update()
        elif self.tool == "shape":
            rect = self._maybe_shape_square(self.start_point, img_pt,
                                             event.modifiers(),
                                             self.mw.shape_kind)
            self.preview_shape = ("shape", rect)
            if self._stroke_preview is not None:
                self._stroke_preview_scaled = None
            self.update()

    def _maybe_square(self, p1, p2, modifiers):
        rect = QRect(p1, p2).normalized()
        if modifiers & Qt.ShiftModifier:
            side = max(rect.width(), rect.height())
            sx = 1 if p2.x() >= p1.x() else -1
            sy = 1 if p2.y() >= p1.y() else -1
            rect = QRect(p1, QPoint(p1.x() + sx * side, p1.y() + sy * side)).normalized()
        return rect

    # ---------- 鼠标抬起 ----------
    def mouseReleaseEvent(self, event):
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        if self.tool == "pan":
            self._panning = False
            self._pan_last = None
            self.setCursor(Qt.OpenHandCursor)
            return
        
        pos = event.pos()
        img_pt = self.widget_to_image(pos)
        use_bg = (event.button() == Qt.RightButton)

        # —— 结束选区整体操作 ——
        if self.sel_mode != SEL_NONE:
            self.sel_mode = SEL_NONE
            self.sel_handle = None
            self.original_parts = None
            self.mw.push_history()
            return

        if not self.drawing:
            self._pending_deselect = False
            return
        self.drawing = False

        if self.tool == "line":
            if self.line_edit:
                if self.line_drag is not None:
                    # 松开控制柄：继续编辑
                    self.line_drag = None
                    self.drawing = False
                    return
                # 拖完基线：进入编辑状态，等用户确认或拖控制点
                self.drawing = False
                if self.line_p0 == self.line_p3:
                    # 单击没拖动 → 取消
                    self.line_edit = False
                    self.preview_shape = None
                    self.refresh()
                    return
            self.refresh()
            return
        
        elif self.tool == "rect":
            rect = self._maybe_square(self.start_point, img_pt, event.modifiers())
            color = self.mw.background if use_bg else self.mw.foreground
            self._make_shape_floating("rect", rect, color)
        elif self.tool == "ellipse":
            rect = self._maybe_square(self.start_point, img_pt, event.modifiers())
            color = self.mw.background if use_bg else self.mw.foreground
            self._make_shape_floating("ellipse", rect, color)
        elif self.tool in ("brush", "eraser"):
            self._smudge_prev = None
            self._smudge_prev_pos = None
            if self.brush_type == "grain_airbrush":
                self._air_timer.stop()
                self._air_hold_pos = None
                self._air_hold_color = None

            if self._stroke_mask is not None:
                # 圆头/方形：把 mask 合成为一张 RGBA，一次性叠到图层
                self._commit_stroke_mask()
            elif self._stroke_preview is not None:
                # 其他笔刷：原逻辑
                layer = self.stack.cur().image
                layer.alpha_composite(self._stroke_preview)
                self._stroke_preview = None

            self.mw.push_history()
            self.refresh()
        elif self.tool == "ellipse_select":
            rect = self._maybe_square(self.start_point, img_pt, event.modifiers())
            ctrl = bool(event.modifiers() & Qt.ControlModifier)
            if rect.width() > 2 and rect.height() > 2:
                # 用椭圆的多边形近似，复用 _extract_floating 的 poly 参数
                poly = self._ellipse_polygon(rect)
                if ctrl and not self.group.is_empty():
                    self._extract_floating(rect, poly, append=True)
                else:
                    self.commit_all()
                    self._extract_floating(rect, poly, append=False)
                self.mw.push_history()
            elif self._pending_deselect and not ctrl:
                self.commit_all()
            self._pending_deselect = False
        elif self.tool == "rect_select":
            rect = self._maybe_square(self.start_point, img_pt, event.modifiers())
            ctrl = bool(event.modifiers() & Qt.ControlModifier)
            if rect.width() > 2 and rect.height() > 2:
                if ctrl and not self.group.is_empty():
                    self._extract_floating(rect, append=True)
                else:
                    self.commit_all()
                    self._extract_floating(rect, append=False)
                self.mw.push_history()
            elif self._pending_deselect and not ctrl:
                self.commit_all()
            self._pending_deselect = False
        elif self.tool == "lasso":
            ctrl = bool(event.modifiers() & Qt.ControlModifier)
            if len(self.lasso_points) >= 3:
                poly = QPolygon(self.lasso_points)
                if ctrl and not self.group.is_empty():
                    self._extract_floating(poly.boundingRect(), poly, append=True)
                else:
                    self.commit_all()
                    self._extract_floating(poly.boundingRect(), poly, append=False)
                self.mw.push_history()
            elif self._pending_deselect and not ctrl:
                self.commit_all()
            self._pending_deselect = False
            self.lasso_points = []
        elif self.tool == "gradient":
            self._apply_gradient(self.start_point, img_pt,
                                 self.mw.grad_type.currentIndex(),
                                 self.mw.foreground, self.mw.background)
            self.mw.push_history()
        elif self.tool == "shape":
            rect = self._maybe_shape_square(self.start_point, img_pt,
                                             event.modifiers(),
                                             self.mw.shape_kind)
            color = self.mw.background if use_bg else self.mw.foreground
            self._make_shape_floating(self.mw.shape_kind, rect, color)

        self.preview_shape = None
        self.refresh()

    # ---------- 绘制实现 ----------
    def _commit_stroke_mask(self):
        if self._stroke_mask is None:
            return
        opacity = self.brush_opacity / 100.0
        r, g, b, _ = self._stroke_color
        alpha = self._stroke_mask
        if opacity < 1.0:
            alpha = alpha.point(lambda v: int(v * opacity))
        rgba = Image.new("RGBA", self._stroke_mask.size, (r, g, b, 255))
        rgba.putalpha(alpha)

        if self.group.is_empty():
            # 无选区：叠到当前图层
            layer = self.stack.cur().image
            if self._stroke_erase:
                inv = alpha.point(lambda v: 255 - v)
                lr, lg, lb, la = layer.split()
                la = ImageChops.multiply(la, inv)
                layer.paste(Image.merge("RGBA", (lr, lg, lb, la)), (0, 0))
            else:
                layer.alpha_composite(rgba)
        else:
            # 有选区：叠到每个 part 的浮动图，用 part 多边形裁剪
            for p in self.group.parts:
                img = p.get("img")
                pos = p.get("pos")
                if img is None or pos is None:
                    continue
                # 整图大小的 mask，只保留该 part 多边形内
                part_mask = Image.new("L", self._stroke_mask.size, 0)
                ImageDraw.Draw(part_mask).polygon(
                    [(q.x(), q.y()) for q in p["poly"]], fill=255)
                clipped = ImageChops.multiply(alpha, part_mask)

                if self._stroke_erase:
                    inv = clipped.point(lambda v: 255 - v)
                    # 把 part 浮动图贴到整图临时层做擦除，再裁回
                    temp = Image.new("RGBA", self._stroke_mask.size, (0, 0, 0, 0))
                    temp.paste(img, (pos.x(), pos.y()))
                    tr, tg, tb, ta = temp.split()
                    ta = ImageChops.multiply(ta, inv)
                    temp = Image.merge("RGBA", (tr, tg, tb, ta))
                    p["img"] = temp.crop(
                        (pos.x(), pos.y(),
                         pos.x() + img.width, pos.y() + img.height))
                else:
                    # 把 rgba 的 alpha 换成 clipped，再叠到 part 浮动图
                    part_rgba = rgba.copy()
                    part_rgba.putalpha(clipped)
                    temp = Image.new("RGBA", self._stroke_mask.size, (0, 0, 0, 0))
                    temp.paste(img, (pos.x(), pos.y()))
                    temp.alpha_composite(part_rgba)
                    p["img"] = temp.crop(
                        (pos.x(), pos.y(),
                         pos.x() + img.width, pos.y() + img.height))
            self.group.recompute_bbox()

        self._stroke_mask = None
        self._stroke_color = None
        self._stroke_erase = False

    def _ellipse_polygon(self, rect):
        """把椭圆矩形转成多边形近似（用于选区 mask）"""
        steps = max(64, int((rect.width() + rect.height()) * 2))
        cx = rect.center().x()
        cy = rect.center().y()
        rx = rect.width() / 2
        ry = rect.height() / 2
        poly = QPolygon()
        for i in range(steps):
            a = 2 * math.pi * i / steps
            poly.append(QPoint(int(cx + rx * math.cos(a)),
                               int(cy + ry * math.sin(a))))
        return poly

    def _hit_test_line_handle(self, widget_pos):
        """返回命中的手柄：'p0' / 'p3' / 'c1' / 'c2' / None"""
        if not self.line_edit:
            return None
        # 控制点优先（曲线模式下），直线模式下不显示控制点
        targets = []
        if self.line_is_curve:
            targets = [("c1", self.line_ctrl1), ("c2", self.line_ctrl2),
                       ("p0", self.line_p0), ("p3", self.line_p3)]
        else:
            targets = [("p0", self.line_p0), ("p3", self.line_p3)]
        for name, pt in targets:
            if pt is None:
                continue
            w = self.image_to_widget(pt)
            if (widget_pos - w).manhattanLength() < 12:
                return name
        return None

    def _commit_line(self):
        if not self.line_edit:
            return
        if self._line_preview_img is None:
            self._rebuild_line_preview()
        if self._line_preview_img is not None:
            if self.group.is_empty():
                self.stack.cur().image.alpha_composite(self._line_preview_img)
            else:
                def do(img):
                    img.alpha_composite(self._line_preview_img)
                    return img
                self._apply_to_layer_in_selection(do)
        self.line_edit = False
        self.line_drag = None
        self.line_p0 = self.line_p3 = None
        self.line_ctrl1 = self.line_ctrl2 = None
        self.preview_shape = None
        self._line_preview_img = None
        self.mw.push_history()
        self.refresh()

    def _styled_segments(self, pts, dash):
        """把折线 pts 按 dash 样式切成 [(x1,y1,x2,y2), ...] 的可视段列表。
        dot 模式返回的是零长度段（单点盖章用）。"""
        if dash == "solid" or len(pts) < 2:
            return [(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                    for i in range(len(pts) - 1)]

        size = self.mw.brush_size
        if dash == "dash":
            on_len = max(1.0, size * 3.0)
            off_len = max(1.0, size * 2.0)
        else:  # dot
            on_len = max(1.0, size * 0.4)
            off_len = max(1.0, size * 2.0)

        segs = []
        drawing_on = True
        acc = 0.0
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if seg_len == 0:
                continue
            t = 0.0
            while t < seg_len:
                remain = (on_len if drawing_on else off_len) - acc
                step = min(remain, seg_len - t)
                if drawing_on:
                    a = t / seg_len
                    b = (t + step) / seg_len
                    segs.append((x1 + (x2-x1)*a, y1 + (y2-y1)*a,
                                 x1 + (x2-x1)*b, y1 + (y2-y1)*b))
                t += step
                acc += step
                if acc >= (on_len if drawing_on else off_len) - 1e-6:
                    acc = 0.0
                    drawing_on = not drawing_on
        return segs

    def _draw_line_object(self, color):
        if self.line_p0 is None or self.line_p3 is None:
            return
        if self.line_is_curve:
            samples = 80
            pts = []
            p0, p1, p2, p3 = (self.line_p0, self.line_ctrl1,
                              self.line_ctrl2, self.line_p3)
            for i in range(samples + 1):
                t = i / samples
                x = ((1-t)**3 * p0.x() + 3*(1-t)**2 * t * p1.x()
                     + 3*(1-t) * t**2 * p2.x() + t**3 * p3.x())
                y = ((1-t)**3 * p0.y() + 3*(1-t)**2 * t * p1.y()
                     + 3*(1-t) * t**2 * p2.y() + t**3 * p3.y())
                pts.append((int(x), int(y)))
        else:
            pts = [(self.line_p0.x(), self.line_p0.y()),
                   (self.line_p3.x(), self.line_p3.y())]

        dash = self.mw.line_dash
        arrow = self.mw.line_arrow

        def do(img):
            for (xa, ya, xb, yb) in self._styled_segments(pts, dash):
                draw_thick_line(img, (xa, ya), (xb, yb),
                                color.getRgb(), self.mw.brush_size,
                                self.mw.brush_hardness, False)
            if arrow in ("end", "both") and len(pts) >= 2:
                _draw_arrow_tip(img, pts[-2], pts[-1], color.getRgb(),
                                self.mw.brush_size, self.mw.brush_hardness)
            if arrow == "both" and len(pts) >= 2:
                _draw_arrow_tip(img, pts[1], pts[0], color.getRgb(),
                                self.mw.brush_size, self.mw.brush_hardness)
            return img

        if self.group.is_empty():
            self.stack.cur().image = do(self.stack.cur().image.copy())
        else:
            self._apply_to_layer_in_selection(do)

    def _maybe_shape_square(self, p1, p2, modifiers, kind):
        """按 Shift 时生成正多边形约束的包围盒"""
        rect = QRect(p1, p2).normalized()
        if not (modifiers & Qt.ShiftModifier):
            return rect
        side = max(rect.width(), rect.height())
        sx = 1 if p2.x() >= p1.x() else -1
        sy = 1 if p2.y() >= p1.y() else -1
        return QRect(p1, QPoint(p1.x() + sx * side,
                                p1.y() + sy * side)).normalized()

    def _effective_stroke_mask(self):
        if self._stroke_mask is None:
            return None
        if self.brush_opacity < 100:
            return self._stroke_mask.point(
                lambda v: int(v * self.brush_opacity / 100))
        return self._stroke_mask
    
    def _stroke(self, p1, p2, color, erase):
        bt = self.brush_type
        size = self.mw.brush_size
        shape = "square" if bt == "square" else "round"
        grain = self.grain_amount if bt in ("grain", "grain_airbrush") else 0.0
        opacity = self.brush_opacity / 100.0
        use_mask = (self._stroke_mask is not None
                    and bt in ("round", "square", "airbrush", "grain"))

        # ---------- 涂抹：保持原逻辑，不走 mask ----------
        if bt == "smudge":
            dist = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            step = max(1.0, self.mw.brush_size * 0.1)
            n = max(1, int(dist / step))
            for i in range(1, n + 1):
                t = i / n
                x = int(p1.x() + (p2.x() - p1.x()) * t)
                y = int(p1.y() + (p2.y() - p1.y()) * t)
                self._smudge_step(x, y)
            self.refresh()
            return

        # ---------- 软喷枪：走 mask ----------
        if bt == "airbrush" and use_mask:
            col = color.getRgb() if color is not None else (0, 0, 0, 255)
            softness = 25
            h = self.mw.brush_hardness / 100.0
            air_opacity = opacity * (0.05 + 0.30 * h)

            def seg(img):
                draw_thick_line(img, (p1.x(), p1.y()), (p2.x(), p2.y()),
                                col, self.mw.brush_size, softness,
                                False,              # ← 始终 False
                                shape="round", grain=0.0, opacity=air_opacity)
            self._stroke_segment_via_mask(seg)
            self.update()
            return

        # ---------- 颗粒喷枪：走 _grain_air_batch ----------
        if bt == "grain_airbrush":
            dist = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            spacing = max(1, size * 0.1)
            n = max(0, int(dist / spacing))
            samples = []
            if n == 0:
                samples.append((int(p1.x()), int(p1.y()), color))
            else:
                for i in range(n + 1):
                    t = i / n
                    x = p1.x() + (p2.x() - p1.x()) * t
                    y = p1.y() + (p2.y() - p1.y()) * t
                    samples.append((int(x), int(y), color))
            self._grain_air_batch(samples)
            self.update()
            return

        # ---------- 颗粒笔：走 mask ----------
        if bt == "grain" and use_mask:
            col = color.getRgb() if color is not None else (0, 0, 0, 255)
            spacing = max(2.0, size * 0.2)
            dist = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            n = max(1, int(dist / spacing))

            def seg(img):
                for i in range(n + 1):
                    t = i / n
                    x = p1.x() + (p2.x() - p1.x()) * t
                    y = p1.y() + (p2.y() - p1.y()) * t
                    stamp_brush(img, x, y, col, size,
                                self.mw.brush_hardness,
                                False,              # ← 始终 False
                                shape=shape, grain=grain * 10, opacity=opacity)
            self._stroke_segment_via_mask(seg)
            self.update()
            return

        # ---------- 圆头 / 方形：走 mask ----------
        if bt in ("round", "square") and use_mask:
            self._accumulate_stroke_mask(p1, p2, shape)
            self.update()
            return

    def _stroke_segment_via_mask(self, brush_fn):
        if self._stroke_mask is None:
            return
        w, h = self.stack.width(), self.stack.height()
        temp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        brush_fn(temp)
        seg_a = temp.split()[3]
        self._stroke_mask = ImageChops.lighter(self._stroke_mask, seg_a)

    def _accumulate_stroke_mask(self, p1, p2, shape):
        """沿线段盖章，把每个章的 alpha 用 ImageChops.lighter 合并到 _stroke_mask。
        同一像素在同一笔内只保留最大覆盖度，避免重复叠加。"""
        if self._stroke_mask is None:
            return
        size = max(1, int(self.mw.brush_size))
        hardness = self.mw.brush_hardness
        stamp = get_brush_stamp(size, hardness, shape)   # numpy uint8 (size,size,4)
        stamp_alpha = Image.fromarray(stamp[..., 3], "L")   # 只取 alpha

        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        dist = math.hypot(x2 - x1, y2 - y1)
        step = max(1.0, size * 0.1)
        n = max(1, int(dist / step))

        W, H = self._stroke_mask.size
        mask = self._stroke_mask
        for i in range(n + 1):
            t = i / n
            cx = int(x1 + (x2 - x1) * t)
            cy = int(y1 + (y2 - y1) * t)
            x0 = cx - size // 2
            y0 = cy - size // 2
            # 与画布求交
            lx0 = max(0, -x0); ly0 = max(0, -y0)
            lx1 = size - max(0, (x0 + size) - W)
            ly1 = size - max(0, (y0 + size) - H)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            rx0 = max(0, x0); ry0 = max(0, y0)
            rx1 = rx0 + (lx1 - lx0); ry1 = ry0 + (ly1 - ly0)
            # 目标区域
            region = mask.crop((rx0, ry0, rx1, ry1))
            # 章的对应子块
            sub = stamp_alpha.crop((lx0, ly0, lx1, ly1))
            # 取 max
            merged = ImageChops.lighter(region, sub)
            mask.paste(merged, (rx0, ry0))
        
    def _smudge_step(self, x, y):
        if self._smudge_prev_pos is None:
            self._smudge_prev_pos = QPoint(x, y)
            return
        px, py = self._smudge_prev_pos.x(), self._smudge_prev_pos.y()
        size = self.mw.brush_size
        half = size // 2

        if not self.group.is_empty():
            # 有选区：涂浮动图
            for p in self.group.parts:
                img = p.get("img")
                pos = p.get("pos")
                if img is None or pos is None:
                    continue
                lx = x - pos.x(); ly = y - pos.y()
                # 完全在浮动图外就跳过
                if (lx + half < 0 or ly + half < 0 or
                        lx - half > img.width or ly - half > img.height):
                    continue
                lpx = px - pos.x(); lpy = py - pos.y()
                smudge_inplace(img, lx, ly, size, self.smudge_strength, lpx, lpy)
        else:
            # 无选区：涂图层
            smudge_inplace(self.stack.cur().image, x, y, size,
                           self.smudge_strength, px, py)

        self._smudge_prev_pos = QPoint(x, y)
        
    def _make_shape_floating(self, kind, rect, color):
        self.commit_all()
        pad = self.mw.brush_size + 2
        w = max(1, rect.width()) + pad * 2
        h = max(1, rect.height()) + pad * 2
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        fill = self.mw.background.getRgb() if self.mw.shape_fill else None

        pts = self._shape_polygon(kind, rect.width(), rect.height(), pad)

        if fill is not None and len(pts) >= 3:
            ImageDraw.Draw(layer).polygon(pts, fill=fill)
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            draw_thick_line(layer, a, b, color.getRgb(),
                            self.mw.brush_size, self.mw.brush_hardness, False)

        poly = QPolygon([rect.topLeft(), rect.topRight(),
                         rect.bottomRight(), rect.bottomLeft()])
        self.group.add_part(poly, QRect(rect), layer,
                            QPoint(rect.left() - pad, rect.top() - pad))
        self.mw.push_history()
        self.refresh()

    def _shape_polygon(self, kind, w, h, pad):
        x0, y0 = pad, pad
        x1, y1 = pad + w, pad + h
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        if kind == "rect":
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if kind == "ellipse":
            rx = (x1 - x0) / 2
            ry = (y1 - y0) / 2
            steps = max(64, int((rx + ry) * 2))
            return [(cx + rx * math.cos(2 * math.pi * i / steps),
                     cy + ry * math.sin(2 * math.pi * i / steps))
                    for i in range(steps)]
        if kind == "triangle":
            return [(cx, y0), (x1, y1), (x0, y1)]
        if kind == "hexagon":
            rx = (x1 - x0) / 2
            ry = (y1 - y0) / 2
            return [(cx + rx * math.cos(math.pi / 3 * i - math.pi / 6),
                     cy + ry * math.sin(math.pi / 3 * i - math.pi / 6))
                    for i in range(6)]
        if kind == "star":
            rx = (x1 - x0) / 2
            ry = (y1 - y0) / 2
            pts = []
            for i in range(10):
                a = -math.pi / 2 + math.pi / 5 * i
                r = 1.0 if i % 2 == 0 else 0.382
                pts.append((cx + rx * r * math.cos(a),
                            cy + ry * r * math.sin(a)))
            return pts
        if kind == "heart":
            rx = (x1 - x0) / 2
            ry = (y1 - y0) / 2
            pts = []
            steps = 120
            for i in range(steps):
                t = 2 * math.pi * i / steps
                hx = 16 * math.sin(t) ** 3
                hy = (13 * math.cos(t) - 5 * math.cos(2 * t)
                      - 2 * math.cos(3 * t) - math.cos(4 * t))
                pts.append((cx + rx * hx / 16, cy - ry * hy / 17))
            return pts
        if kind == "pentagon":
            rx = (x1 - x0) / 2
            ry = (y1 - y0) / 2
            pts = []
            for i in range(5):
                a = -math.pi / 2 + 2 * math.pi / 5 * i
                pts.append((cx + rx * math.cos(a),
                            cy + ry * math.sin(a)))
            return pts
        if kind == "octagon":
            rx = (x1 - x0) / 2
            ry = (y1 - y0) / 2
            pts = []
            for i in range(8):
                a = -math.pi / 2 + 2 * math.pi / 8 * i
                pts.append((cx + rx * math.cos(a),
                            cy + ry * math.sin(a)))
            return pts
        if kind == "diamond":
            return [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
        if kind == "four_arc_star":
            rx = (x1 - x0) / 2
            ry = (y1 - y0) / 2
            tips = [
                (cx, cy - ry),   # 上
                (cx + rx, cy),   # 右
                (cx, cy + ry),   # 下
                (cx - rx, cy),   # 左
            ]
            concave = 0.45   # 0.0~1.0，越小凹得越深；0.45 接近图示
            pts = []
            n = 4
            for i in range(n):
                t1 = tips[i]
                t2 = tips[(i + 1) % n]
                mx = (t1[0] + t2[0]) / 2
                my = (t1[1] + t2[1]) / 2
                cpx = cx + (mx - cx) * concave
                cpy = cy + (my - cy) * concave
                steps = 40
                for s in range(steps):
                    u = s / steps
                    bx = (1-u)**2 * t1[0] + 2*(1-u)*u * cpx + u**2 * t2[0]
                    by = (1-u)**2 * t1[1] + 2*(1-u)*u * cpy + u**2 * t2[1]
                    pts.append((bx, by))
            return pts
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _draw_line_with_style(self, p1, p2, color):
        """直线：支持虚线/点线/箭头"""
        def do(img):
            draw_styled_line(img, (p1.x(), p1.y()), (p2.x(), p2.y()),
                             color.getRgb(), self.mw.brush_size,
                             self.mw.brush_hardness,
                             self.mw.line_dash, self.mw.line_arrow)
            return img
        self._apply_to_layer_in_selection(do)

    def _draw_curve_with_style(self, points, color):
        """曲线：以采样点作为控制点，用样条平滑"""
        if len(points) < 2:
            return
        # 简化：直接用折线，但用 draw_thick_line 连接每个相邻点
        def do(img):
            dash = self.mw.line_dash
            arrow = self.mw.line_arrow
            # 把折线离散成多段，按 dash 决定是否跳过
            segs = self._polyline_points(points)
            if dash == "solid":
                for i in range(len(segs) - 1):
                    draw_thick_line(img, segs[i], segs[i + 1],
                                    color.getRgb(), self.mw.brush_size,
                                    self.mw.brush_hardness, False)
            else:
                # 虚线：按累计长度控制开/关
                on_len = self.mw.brush_size * 2 if dash == "dash" else self.mw.brush_size
                off_len = on_len
                acc = 0.0
                drawing_on = True
                for i in range(len(segs) - 1):
                    x1, y1 = segs[i]; x2, y2 = segs[i + 1]
                    seg_len = math.hypot(x2 - x1, y2 - y1)
                    t = 0.0
                    while t < seg_len:
                        step = min((on_len if drawing_on else off_len) - acc,
                                   seg_len - t)
                        if drawing_on:
                            xx1 = x1 + (x2 - x1) * (t / seg_len)
                            yy1 = y1 + (y2 - y1) * (t / seg_len)
                            xx2 = x1 + (x2 - x1) * ((t + step) / seg_len)
                            yy2 = y1 + (y2 - y1) * ((t + step) / seg_len)
                            draw_thick_line(img, (xx1, yy1), (xx2, yy2),
                                            color.getRgb(), self.mw.brush_size,
                                            self.mw.brush_hardness, False)
                        t += step
                        acc += step
                        if acc >= (on_len if drawing_on else off_len) - 1e-6:
                            acc = 0.0
                            drawing_on = not drawing_on
            # 箭头
            if arrow in ("end", "both"):
                self._draw_arrow(img, segs[-2], segs[-1], color)
            if arrow == "both":
                self._draw_arrow(img, segs[1], segs[0], color)
            return img
        self._apply_to_layer_in_selection(do)

    def _polyline_points(self, points):
        """把 QPoint 列表转成 (x,y) 元组列表，并可在此加样条平滑"""
        return [(p.x(), p.y()) for p in points]

    def _draw_arrow(self, img, from_pt, tip_pt, color):
        x1, y1 = from_pt
        x2, y2 = tip_pt
        angle = math.atan2(y2 - y1, x2 - x1)
        size = max(8, self.mw.brush_size * 3)
        left = (x2 - size * math.cos(angle - math.pi / 6),
                y2 - size * math.sin(angle - math.pi / 6))
        right = (x2 - size * math.cos(angle + math.pi / 6),
                 y2 - size * math.sin(angle + math.pi / 6))
        draw_thick_line(img, (x2, y2), left, color.getRgb(),
                        self.mw.brush_size, self.mw.brush_hardness, False)
        draw_thick_line(img, (x2, y2), right, color.getRgb(),
                        self.mw.brush_size, self.mw.brush_hardness, False)

    # ---------- 文字 ----------
    def _insert_text(self, img_pt):
        dlg = TextDialog(self, self.mw.foreground)
        if dlg.exec_() != QDialog.Accepted:
            return
        text = dlg.text_edit.text()
        if not text:
            return
        font_path = dlg.font_path
        font_size = dlg.size_spin.value()
        color = dlg.color
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        bbox = font.getbbox(text)
        tw = max(1, bbox[2] - bbox[0]) + 6
        th = max(1, bbox[3] - bbox[1]) + 6
        layer = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.text((3 - bbox[0], 3 - bbox[1]), text, font=font, fill=color.getRgb())

        self.commit_all()
        cx = img_pt.x() - tw // 2
        cy = img_pt.y() - th // 2
        poly = QPolygon([
            QPoint(cx, cy), QPoint(cx + tw, cy),
            QPoint(cx + tw, cy + th), QPoint(cx, cy + th)])
        self.group.add_part(poly, QRect(cx, cy, tw, th), layer, QPoint(cx, cy))
        self.float_meta = {"text": text, "font_path": font_path,
                           "size": font_size, "color": color}
        self.mw.push_history()
        self.refresh()

    # ---------- 选区抠出 ----------
    def _extract_floating(self, bbox, poly=None, append=False):
        layer_img = self.stack.cur().image
        x0 = max(0, bbox.left()); y0 = max(0, bbox.top())
        x1 = min(layer_img.width, bbox.right() + 1)
        y1 = min(layer_img.height, bbox.bottom() + 1)
        if x1 <= x0 or y1 <= y0:
            return
        region = layer_img.crop((x0, y0, x1, y1))
        if poly is not None:
            mask = Image.new("L", region.size, 0)
            md = ImageDraw.Draw(mask)
            md.polygon([(p.x() - x0, p.y() - y0) for p in poly], fill=255)
            # 关键修复：与原有 alpha 相乘，已透明处保持透明
            orig_a = region.split()[3]
            new_a = ImageChops.multiply(orig_a, mask)
            region.putalpha(new_a)
        # 原位置清空
        d = ImageDraw.Draw(layer_img)
        if poly is not None:
            d.polygon([(p.x(), p.y()) for p in poly], fill=(0, 0, 0, 0))
        else:
            d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(0, 0, 0, 0))

        p = poly if poly is not None else QPolygon([
            bbox.topLeft(), bbox.topRight(), bbox.bottomRight(), bbox.bottomLeft()])

        if not append:
            self.group.clear()
        self.group.add_part(p, bbox, region, QPoint(x0, y0))
        self.refresh()

    def _merge_into_group_floating(self, region, pos):
        g = self.group
        cur = g.floating
        cur_pos = g.floating_pos
        if cur is None:
            g.floating = region
            g.floating_pos = pos
            return
        x0 = min(cur_pos.x(), pos.x())
        y0 = min(cur_pos.y(), pos.y())
        x1 = max(cur_pos.x() + cur.width, pos.x() + region.width)
        y1 = max(cur_pos.y() + cur.height, pos.y() + region.height)
        new = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
        new.alpha_composite(cur, (cur_pos.x() - x0, cur_pos.y() - y0))
        new.alpha_composite(region, (pos.x() - x0, pos.y() - y0))
        g.floating = new
        g.floating_pos = QPoint(x0, y0)

    # ---------- 整体缩放 / 旋转 ----------
    def _apply_scale_group(self, pos, modifiers):
        if self.original_parts is None:
            return
        orig_bbox = self._bbox_from_parts(self.original_parts)
        if orig_bbox is None or orig_bbox.width() == 0 or orig_bbox.height() == 0:
            return
        new_bbox = QRect(orig_bbox)
        h = self.sel_handle
        p = self.widget_to_image(pos)
        if h in (0, 3, 5): new_bbox.setLeft(p.x())
        if h in (2, 4, 7): new_bbox.setRight(p.x())
        if h in (0, 1, 2): new_bbox.setTop(p.y())
        if h in (5, 6, 7): new_bbox.setBottom(p.y())
        new_bbox = new_bbox.normalized()
        if new_bbox.width() < 2 or new_bbox.height() < 2:
            return

        if modifiers & Qt.ShiftModifier:
            sx = new_bbox.width() / orig_bbox.width()
            sy = new_bbox.height() / orig_bbox.height()
            s = max(abs(sx), abs(sy))
            ax = orig_bbox.center().x()
            ay = orig_bbox.center().y()
            nw = int(orig_bbox.width() * s); nh = int(orig_bbox.height() * s)
            new_bbox = QRect(int(ax - nw / 2), int(ay - nh / 2), nw, nh)

        sx = new_bbox.width() / orig_bbox.width()
        sy = new_bbox.height() / orig_bbox.height()
        ox, oy = orig_bbox.left(), orig_bbox.top()

        new_parts = []
        for p in self.original_parts:
            poly = QPolygon()
            for q in p["poly"]:
                nx = new_bbox.left() + (q.x() - ox) * sx
                ny = new_bbox.top() + (q.y() - oy) * sy
                poly.append(QPoint(int(nx), int(ny)))
            new_p = {
                "poly": poly,
                "bbox": poly.boundingRect(),
                "img": None, "pos": None,
            }
            if p.get("img") is not None:
                fw = max(1, int(p["img"].width * sx))
                fh = max(1, int(p["img"].height * sy))
                new_p["img"] = p["img"].resize((fw, fh), Image.LANCZOS)
                # 位置：相对旧 bbox 缩放映射
                npx = new_bbox.left() + (p["pos"].x() - ox) * sx
                npy = new_bbox.top() + (p["pos"].y() - oy) * sy
                new_p["pos"] = QPoint(int(npx), int(npy))
            new_parts.append(new_p)
        self.group.parts = new_parts
        self.group.recompute_bbox()

    def _apply_rotate_group(self, pos):
        if (self.rotate_center is None or self.rotate_start_angle is None
                or self.original_parts is None):
            return
        cur = math.atan2(pos.y() - self.rotate_center.y(),
                         pos.x() - self.rotate_center.x())
        delta = math.degrees(cur - self.rotate_start_angle)
        pil_angle = -delta

        cx, cy = self.rotate_center.x(), self.rotate_center.y()
        rad = math.radians(delta)
        ca, sa = math.cos(rad), math.sin(rad)
        new_parts = []
        for p in self.original_parts:
            poly = QPolygon()
            for q in p["poly"]:
                dx, dy = q.x() - cx, q.y() - cy
                poly.append(QPoint(int(dx * ca - dy * sa + cx),
                                   int(dx * sa + dy * ca + cy)))
            new_p = {"poly": poly, "bbox": poly.boundingRect(),
                     "img": None, "pos": None}
            if p.get("img") is not None:
                rotated = p["img"].rotate(pil_angle, resample=Image.BICUBIC,
                                          expand=True)
                # 原 part 中心旋转后仍是新图中心
                pcx = p["pos"].x() + p["img"].width / 2
                pcy = p["pos"].y() + p["img"].height / 2
                ndx = pcx - cx; ndy = pcy - cy
                ncx = ndx * ca - ndy * sa + cx
                ncy = ndx * sa + ndy * ca + cy
                new_p["img"] = rotated
                new_p["pos"] = QPoint(int(ncx - rotated.width / 2),
                                      int(ncy - rotated.height / 2))
            new_parts.append(new_p)
        self.group.parts = new_parts
        self.group.recompute_bbox()

    def _bbox_from_parts(self, parts):
        if not parts:
            return None
        r = QRect(parts[0]["bbox"])
        for p in parts[1:]:
            r = r.united(p["bbox"])
        return r

    # ---------- 提交 / 清空 ----------
    def commit_all(self):
        target = self.stack.cur().image
        for part in self.group.parts:
            if part.get("img") is not None and part.get("pos") is not None:
                target.alpha_composite(
                    part["img"],
                    (int(part["pos"].x()), int(part["pos"].y())))
        self.group.clear()
        self.sel_mode = SEL_NONE
        self.refresh()

    # ---------- 取色 / 填充 ----------
    def pick_color(self, widget_pos, secondary=False):
        img_pt = self.widget_to_image(widget_pos)
        img = self.stack.flatten()
        x, y = img_pt.x(), img_pt.y()
        if 0 <= x < img.width and 0 <= y < img.height:
            r, g, b, a = img.getpixel((x, y))
            color = QColor(r, g, b, a)
            if secondary:
                self.mw.set_background(color)
            else:
                self.mw.set_foreground(color)

    def flood_fill(self, layer_img, pt, color):
        x, y = pt.x(), pt.y()
        if not (0 <= x < layer_img.width and 0 <= y < layer_img.height):
            return
        tolerance = self.mw.fill_tolerance
        from collections import deque
        target = layer_img.getpixel((x, y))
        fill = color.getRgb()
        if target == fill:
            return
        w, h = layer_img.size
        visited = bytearray(w * h)
        dq = deque([(x, y)])
        px = layer_img.load()

        def close(c1, c2):
            return (abs(c1[0] - c2[0]) <= tolerance and
                    abs(c1[1] - c2[1]) <= tolerance and
                    abs(c1[2] - c2[2]) <= tolerance and
                    abs(c1[3] - c2[3]) <= tolerance)

        while dq:
            cx, cy = dq.popleft()
            idx = cy * w + cx
            if visited[idx]:
                continue
            visited[idx] = 1
            if not close(px[cx, cy], target):
                continue
            px[cx, cy] = fill
            if cx > 0: dq.append((cx - 1, cy))
            if cx < w - 1: dq.append((cx + 1, cy))
            if cy > 0: dq.append((cx, cy - 1))
            if cy < h - 1: dq.append((cx, cy + 1))

    # ---------- 键盘 ----------
    def keyPressEvent(self, event):
        mods = event.modifiers()
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selection()
        elif event.key() == Qt.Key_Escape:
            if self.line_edit:
                self._commit_line()
                return
            had_sel = not self.group.is_empty()
            self.commit_all()
            if had_sel:
                self.mw.push_history()
            self.update()
        elif mods & Qt.ControlModifier and event.key() == Qt.Key_C:
            self.copy_selection()
        elif mods & Qt.ControlModifier and event.key() == Qt.Key_V:
            self.mw.paste_from_clipboard()
        elif mods & Qt.ControlModifier and event.key() == Qt.Key_X:
            self.cut_selection()
        elif mods & Qt.ControlModifier and event.key() == Qt.Key_A:
            self.select_all()
        elif mods & Qt.ControlModifier and event.key() == Qt.Key_D:
            self.deselect()
        elif (mods & Qt.ControlModifier and mods & Qt.ShiftModifier
              and event.key() == Qt.Key_I):
            self.invert_selection()
        else:
            super().keyPressEvent(event)

    def delete_selection(self):
        if self.group.is_empty():
            return
        self.group.clear()
        self.mw.push_history()
        self.refresh()

    def copy_selection(self):
        if self.group.is_empty():
            return
        # 以整体 bbox 为画布，合并所有 part
        bbox = self.group.group_bbox
        canvas = Image.new("RGBA", (bbox.width(), bbox.height()), (0, 0, 0, 0))
        for p in self.group.parts:
            if p.get("img") is not None:
                canvas.alpha_composite(
                    p["img"],
                    (p["pos"].x() - bbox.left(), p["pos"].y() - bbox.top()))
        self.mw.clipboard = canvas
        self.mw.statusBar().showMessage("已复制", 1500)

    def cut_selection(self):
        if self.group.is_empty():
            return
        self.copy_selection()
        self.group.clear()
        self.mw.push_history()
        self.refresh()

    def paste_clipboard(self):
        if self.mw.clipboard is None:
            return
        self.commit_all()
        img = self.mw.clipboard.copy()
        cx = max(0, (self.stack.width() - img.width) // 2)
        cy = max(0, (self.stack.height() - img.height) // 2)
        poly = QPolygon([
            QPoint(cx, cy), QPoint(cx + img.width, cy),
            QPoint(cx + img.width, cy + img.height),
            QPoint(cx, cy + img.height)])
        self.group.add_part(poly, QRect(cx, cy, img.width, img.height),
                            img, QPoint(cx, cy))
        self.refresh()

    # ---------- 变换 / 特效 ----------
    def apply_transform(self, kind):
        """kind: rot90cw / rot90ccw / rot180 / flipH / flipV"""
        PIL_OPS = {
            "rot90cw": lambda im: im.transpose(Image.ROTATE_270),
            "rot90ccw": lambda im: im.transpose(Image.ROTATE_90),
            "rot180": lambda im: im.transpose(Image.ROTATE_180),
            "flipH": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
            "flipV": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
        }
        op = PIL_OPS[kind]

        if not self.group.is_empty():
            # 有选区：对每个 part 的浮动图分别变换
            for p in self.group.parts:
                img = p.get("img")
                pos = p.get("pos")
                if img is None or pos is None:
                    continue
                new = op(img)
                # 保持原 part 中心不变
                cx = pos.x() + img.width / 2
                cy = pos.y() + img.height / 2
                p["img"] = new
                p["pos"] = QPoint(int(cx - new.width / 2),
                                  int(cy - new.height / 2))
                # 更新该 part 的多边形为其新外接矩形
                bbox = QRect(p["pos"].x(), p["pos"].y(),
                             new.width, new.height)
                p["bbox"] = bbox
                p["poly"] = QPolygon([bbox.topLeft(), bbox.topRight(),
                                      bbox.bottomRight(), bbox.bottomLeft()])
            self.group.recompute_bbox()
            self.refresh()
        else:
            # 无选区：对当前图层整体变换
            layer = self.stack.cur()
            layer.image = op(layer.image)
            self.refresh()

    def apply_effect(self, kind, parent=None):
        def do_effect(im, params):
            if kind == "sharpen":
                return im.filter(ImageFilter.UnsharpMask(
                    radius=params.get("radius", 2),
                    percent=params.get("percent", 150),
                    threshold=params.get("threshold", 3)))
            if kind == "mosaic":
                return self._mosaic(im, params.get("block", 8))
            if kind == "blur":
                return im.filter(ImageFilter.GaussianBlur(
                    radius=params.get("radius", 2)))
            if kind == "gray":
                return im.convert("L").convert("RGBA")
            return im

        params = self._ask_effect_params(kind, parent)
        if params is None:
            return

        if not self.group.is_empty():
            # 有选区：每个 part 的浮动图分别处理
            for p in self.group.parts:
                if p.get("img") is not None:
                    p["img"] = do_effect(p["img"], params)
            self.refresh()
        else:
            # 无选区：对当前图层整体处理
            self.stack.cur().image = do_effect(self.stack.cur().image, params)
            self.refresh()

    def _mosaic(self, im, block):
        block = max(1, int(block))
        w, h = im.size
        small = im.resize((max(1, w // block), max(1, h // block)), Image.BOX)
        return small.resize((w, h), Image.NEAREST)

    def _ask_effect_params(self, kind, parent):
        dlg = QDialog(parent)
        dlg.setWindowTitle({
            "sharpen": "锐化", "mosaic": "马赛克",
            "blur": "模糊", "gray": "灰度",
        }.get(kind, "特效"))
        form = QFormLayout(dlg)
        spins = {}
        if kind == "sharpen":
            for key, label, lo, hi, val in [
                ("radius", "半径", 1, 20, 2),
                ("percent", "强度%", 10, 500, 150),
                ("threshold", "阈值", 0, 50, 3)]:
                s = QSpinBox(); s.setRange(lo, hi); s.setValue(val)
                form.addRow(label, s); spins[key] = s
        elif kind == "mosaic":
            s = QSpinBox(); s.setRange(1, 200); s.setValue(8)
            form.addRow("块大小", s); spins["block"] = s
        elif kind == "blur":
            s = QSpinBox(); s.setRange(1, 100); s.setValue(2)
            form.addRow("半径", s); spins["radius"] = s

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return {k: s.value() for k, s in spins.items()}
    
    # ---------- 选区便捷操作 ----------
    def select_all(self):
        self.commit_all()
        w, h = self.stack.width(), self.stack.height()
        layer_img = self.stack.cur().image
        region = layer_img.copy()
        self.stack.cur().image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bbox = QRect(0, 0, w, h)
        poly = QPolygon([bbox.topLeft(), bbox.topRight(),
                         bbox.bottomRight(), bbox.bottomLeft()])
        self.group.clear()
        self.group.add_part(poly, bbox, region, QPoint(0, 0))
        self.refresh()

    def deselect(self):
        had_sel = not self.group.is_empty()
        self.commit_all()
        if had_sel:
            self.mw.push_history()
        self.update()

    def invert_selection(self):
        if self.group.is_empty():
            # 没有选区时反选 = 全选
            self.select_all(); return
        w, h = self.stack.width(), self.stack.height()
        # 用 mask 图求反：先画所有 parts 为 255，再全图填 255 后相减
        mask = Image.new("L", (w, h), 255)
        d = ImageDraw.Draw(mask)
        for p in self.group.parts:
            d.polygon([(q.x(), q.y()) for q in p["poly"]], fill=0)
        # 从反选 mask 里抠出浮动内容
        self.commit_all()
        layer_img = self.stack.cur().image
        region = layer_img.copy()
        region.putalpha(mask)
        # 原图整层清空
        self.stack.cur().image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bbox = QRect(0, 0, w, h)
        # 用 mask 生成一个多边形 parts 无法表达非凸形状，这里作为一个 part 用外接 bbox
        poly = QPolygon([bbox.topLeft(), bbox.topRight(),
                         bbox.bottomRight(), bbox.bottomLeft()])
        self.group.clear()
        self.group.add_part(poly, bbox, region, QPoint(0, 0))
        self.refresh()

    def _apply_gradient(self, p1, p2, kind, c1, c2):
        w, h = self.stack.width(), self.stack.height()
        if p1 == p2:
            return
        yy, xx = np.mgrid[0:h, 0:w]
        if kind == 0:
            dx = p2.x() - p1.x(); dy = p2.y() - p1.y()
            denom = dx * dx + dy * dy
            if denom == 0:
                return
            t = ((xx - p1.x()) * dx + (yy - p1.y()) * dy) / denom
        else:
            r = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            if r == 0:
                return
            t = np.sqrt((xx - p1.x()) ** 2 + (yy - p1.y()) ** 2) / r
        t = np.clip(t, 0, 1)
        r1, g1, b1 = c1.red(), c1.green(), c1.blue()
        r2, g2, b2 = c2.red(), c2.green(), c2.blue()
        rr = (r1 + (r2 - r1) * t).astype(np.uint8)
        gg = (g1 + (g2 - g1) * t).astype(np.uint8)
        bb = (b1 + (b2 - b1) * t).astype(np.uint8)
        aa = np.full_like(rr, 255, dtype=np.uint8)
        grad = Image.fromarray(np.stack([rr, gg, bb, aa], axis=-1), "RGBA")

        def do(img):
            img.alpha_composite(grad)
            return img
        self._apply_to_layer_in_selection(do)

    def has_selection(self):
        return not self.group.is_empty()

    def _selection_mask_l(self):
        """返回整图大小的 L mask，选区为 255，其他 0；无选区返回 None"""
        if not self.has_selection():
            return None
        w, h = self.stack.width(), self.stack.height()
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        for p in self.group.parts:
            d.polygon([(q.x(), q.y()) for q in p["poly"]], fill=255)
        return mask

    def _apply_to_layer_in_selection(self, func):
        """在当前图层上、只对选区内的像素应用 func(img)->img；
        若已有浮动内容，则绘制作用于浮动内容，保持与画布显示一致"""
        # 无选区：整层处理
        if self.group.is_empty():
            if self._stroke_preview is not None:
                # 绘制中：作用于笔画缓存
                self._stroke_preview = func(self._stroke_preview.copy())
            else:
                self.stack.cur().image = func(self.stack.cur().image.copy())
            return

        # 有选区：对每个 part 的浮动图分别处理
        full_size = self.stack.cur().image.size
        for p in self.group.parts:
            if p.get("img") is None or p.get("pos") is None:
                continue
            img = p["img"]
            pos = p["pos"]
            # 把浮动图贴到一张整图大小的临时层，让 func 能用整图坐标
            temp = Image.new("RGBA", full_size, (0, 0, 0, 0))
            temp.paste(img, (pos.x(), pos.y()), img)
            new_temp = func(temp.copy())
            # 用 part 的多边形 mask 限定只改选区内（支持套索）
            mask = Image.new("L", full_size, 0)
            ImageDraw.Draw(mask).polygon(
                [(q.x(), q.y()) for q in p["poly"]], fill=255)
            composited = Image.composite(new_temp, temp, mask)
            # 裁回浮动区域
            p["img"] = composited.crop(
                (pos.x(), pos.y(),
                 pos.x() + img.width, pos.y() + img.height))
        self.refresh()
        self._notify_thumb()

# ================= 滚动容器 =================
class CanvasScrollArea(QScrollArea):
    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        self.setWidget(canvas)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.horizontalScrollBar().valueChanged.connect(self._sync_rulers)
        self.verticalScrollBar().valueChanged.connect(self._sync_rulers)

    def wheelEvent(self, event):
        # ... 保持你原来的 Ctrl+滚轮缩放逻辑不变
        if event.modifiers() & Qt.ControlModifier:
            vp_pos = event.pos()
            old_zoom = self.canvas.zoom
            img_x = (self.horizontalScrollBar().value() + vp_pos.x()) / old_zoom
            img_y = (self.verticalScrollBar().value() + vp_pos.y()) / old_zoom
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1 / 1.15
            new_zoom = max(0.05, min(16.0, old_zoom * factor))
            if abs(new_zoom - old_zoom) < 1e-6:
                event.accept(); return
            self.canvas.zoom = new_zoom
            self.canvas.refresh_view()
            self.horizontalScrollBar().setValue(int(img_x * new_zoom - vp_pos.x()))
            self.verticalScrollBar().setValue(int(img_y * new_zoom - vp_pos.y()))
            event.accept()
        else:
            super().wheelEvent(event)
        self._sync_rulers()

    def _sync_rulers(self):
        mw = self.canvas.mw
        if hasattr(mw, "h_ruler"):
            mw.h_ruler.set_offset(self.horizontalScrollBar().value())
        if hasattr(mw, "v_ruler"):
            mw.v_ruler.set_offset(self.verticalScrollBar().value())

# ================= 文字对话框 =================

class TextDialog(QDialog):
    def __init__(self, parent, default_color):
        super().__init__(parent)
        self.setWindowTitle("插入文字")
        form = QFormLayout(self)
        self.text_edit = QLineEdit()
        self.font_combo = QFontComboBox()
        self.size_spin = QSpinBox(); self.size_spin.setRange(6, 500); self.size_spin.setValue(36)
        self.color = default_color
        self.color_label = QLabel(); self.color_label.setFixedSize(24, 24)
        self._update_color_label()
        pick = QPushButton("选择颜色")
        pick.clicked.connect(self.pick_color)
        row = QHBoxLayout(); row.addWidget(pick); row.addWidget(self.color_label)

        form.addRow("文字", self.text_edit)
        form.addRow("字体", self.font_combo)
        form.addRow("字号", self.size_spin)
        form.addRow("颜色", row)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _update_color_label(self):
        self.color_label.setStyleSheet(
            f"background-color: {self.color.name()}; border: 1px solid #333;")

    def pick_color(self):
        c = QColorDialog.getColor(self.color, self, "文字颜色")
        if c.isValid():
            self.color = c
            self._update_color_label()

    @property
    def font_path(self):
        """返回字体文件的绝对路径"""
        family = self.font_combo.currentFont().family()
        # Windows 常见字体名 -> 文件名映射
        mapping = {
            "SimSun": "simsun.ttc", "宋体": "simsun.ttc",
            "SimHei": "simhei.ttf", "黑体": "simhei.ttf",
            "Microsoft YaHei": "msyh.ttc", "微软雅黑": "msyh.ttc",
            "KaiTi": "simkai.ttf", "楷体": "simkai.ttf",
            "Arial": "arial.ttf", "Times New Roman": "times.ttf",
            "Courier New": "cour.ttf",
        }
        fname = mapping.get(family, None)
        if fname:
            p = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", fname)
            if os.path.exists(p):
                return p
        # 回退：从系统字体目录按 family 模糊找
        fonts_dir = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
        if os.path.isdir(fonts_dir):
            for f in os.listdir(fonts_dir):
                if f.lower().endswith((".ttf", ".ttc", ".otf")):
                    if family.lower().replace(" ", "") in f.lower().replace(" ", ""):
                        return os.path.join(fonts_dir, f)
        # 最终回退
        return "C:/Windows/Fonts/simhei.ttf"

# ================= 图层对话框 =================

class LayerPanel(QWidget):
    """右侧常驻图层面板"""
    def __init__(self, main_window, stack):
        super().__init__()
        self.mw = main_window
        self.stack = stack
        self._updating = False
        # 提前保存 main_window 引用，供按钮转发方法使用
        self._main_window = main_window
        self._multiframe = False
        self._frames = []
        self._frame_index = 0

        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        self.list = QListWidget()
        self.list.setIconSize(QSize(64, 48))
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.currentRowChanged.connect(self.on_row_changed)
        self.list.itemChanged.connect(self.on_item_changed)
        self.list.itemDoubleClicked.connect(self.on_double_click)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        v.addWidget(self.list, 1)

        # ---------- 刷新按钮 ----------
        row0 = QHBoxLayout()
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self._on_refresh_clicked)   # 改为转发方法
        row0.addWidget(btn_refresh)
        v.addLayout(row0)

        # ---------- 帧模式下仍可用的按钮 ----------
        # 新建 / 删除 / 上移 / 下移 / 刷新：帧模式下走帧逻辑，图层模式下走图层逻辑
        self._frame_ok_buttons = [btn_refresh]
        for text, fn in [("新建", self._on_new_clicked),
                         ("删除", self._on_del_clicked),
                         ("上移", self._on_up_clicked),
                         ("下移", self._on_down_clicked)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            v.addWidget(b)
            self._frame_ok_buttons.append(b)

        # ---------- 帧模式下应禁用的按钮 ----------
        self._layer_only_buttons = []

        b_rename = QPushButton("重命名")
        b_rename.clicked.connect(self.rename_layer)
        v.addWidget(b_rename)
        self._layer_only_buttons.append(b_rename)

        b_merge = QPushButton("合并可见")
        b_merge.clicked.connect(self.merge_visible)
        v.addWidget(b_merge)
        self._layer_only_buttons.append(b_merge)

        b_merge_sel = QPushButton("合并选择")
        b_merge_sel.clicked.connect(self.merge_selected)
        v.addWidget(b_merge_sel)
        self._layer_only_buttons.append(b_merge_sel)

        # ---------- 不透明度 ----------
        row_op = QHBoxLayout()
        row_op.addWidget(QLabel("不透明度"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        row_op.addWidget(self.opacity_slider, 1)
        v.addLayout(row_op)

        # ---------- 混合模式 ----------
        row_bl = QHBoxLayout()
        row_bl.addWidget(QLabel("混合模式"))
        self.blend_combo = QComboBox()
        self.blend_combo.addItems(["正常", "正片叠底", "滤色", "叠加"])
        self.blend_combo.currentIndexChanged.connect(self.on_blend_changed)
        row_bl.addWidget(self.blend_combo, 1)
        v.addLayout(row_bl)

        # ---------- 蒙版按钮 ----------
        row_mk = QHBoxLayout()
        btn_mask_add = QPushButton("添加/编辑蒙版")
        btn_mask_add.clicked.connect(self.edit_mask)
        btn_mask_del = QPushButton("删除蒙版")
        btn_mask_del.clicked.connect(self.remove_mask)
        row_mk.addWidget(btn_mask_add)
        row_mk.addWidget(btn_mask_del)
        v.addLayout(row_mk)
        self._layer_only_buttons.extend([btn_mask_add, btn_mask_del])

        # ---------- 初始化列表 ----------
        self.refresh_list()

    def _on_refresh_clicked(self):
        if self._multiframe and self._main_window is not None:
            self._main_window._update_frame_list()
        else:
            self.refresh_list()
    def _on_new_clicked(self):
        if self._multiframe and self._main_window is not None:
            self._main_window._add_frame()
        else:
            self.add_layer()
    def _on_del_clicked(self):
        if self._multiframe and self._main_window is not None:
            self._main_window._del_frame()
        else:
            self.del_layer()
    def _on_up_clicked(self):
        if self._multiframe and self._main_window is not None:
            self._main_window.switch_frame(self._main_window._frame_index - 1)
        else:
            self.move_up()
    def _on_down_clicked(self):
        if self._multiframe and self._main_window is not None:
            self._main_window.switch_frame(self._main_window._frame_index + 1)
        else:
            self.move_down()

    def refresh_frame_thumb(self, idx, frame_img):
        """只更新某一帧那一行的缩略图（帧模式下用）"""
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.data(Qt.UserRole) == idx:
                thumb = frame_img.copy()
                thumb.thumbnail((64, 48), Image.LANCZOS)
                q = pil_to_qimage(thumb)
                pm = QPixmap.fromImage(q)
                item.setIcon(QIcon(pm))
                break

    def _make_thumb(self, layer_img, layer_idx=None):
        w, h = layer_img.size
        if w == 0 or h == 0:
            return QIcon()
        base = layer_img.copy().convert("RGBA")
        if layer_idx is not None and layer_idx == self.stack.current:
            c = self.mw.canvas
            if not c.group.is_empty():
                for p in c.group.parts:
                    img = p.get("img")
                    pos = p.get("pos")
                    if img is not None and pos is not None:
                        # 用 alpha_composite 叠加，正确处理透明
                        base.alpha_composite(img, (int(pos.x()), int(pos.y())))
        thumb = base.copy()
        thumb.thumbnail((64, 48), Image.LANCZOS)
        q = pil_to_qimage(thumb)
        pm = QPixmap.fromImage(q)
        return QIcon(pm)

    def set_multiframe(self, is_mf, main_window):
        self._multiframe = is_mf
        self._main_window = main_window
        # 帧模式下禁用图层专属控件
        for w in (self.opacity_slider, self.blend_combo):
            w.setEnabled(not is_mf)
        for b in getattr(self, "_layer_only_buttons", []):
            b.setEnabled(not is_mf)
        # 帧模式下保留可用的按钮（显式启用，避免之前被禁用后没恢复）
        for b in getattr(self, "_frame_ok_buttons", []):
            b.setEnabled(True)

    def populate_frames(self, frames, current):
        self._updating = True
        self.list.blockSignals(True)
        self.list.clear()
        for i, fr in enumerate(frames):
            thumb = fr.copy()
            thumb.thumbnail((64, 48), Image.LANCZOS)
            q = pil_to_qimage(thumb)
            pm = QPixmap.fromImage(q)
            item = QListWidgetItem(f"帧 {i + 1}")
            item.setIcon(QIcon(pm))
            item.setData(Qt.UserRole, i)
            self.list.addItem(item)
        if 0 <= current < self.list.count():
            self.list.setCurrentRow(current)
        self.list.blockSignals(False)
        self._updating = False

    def refresh_list(self):
        self._updating = True
        self.list.blockSignals(True)
        self.list.clear()
        for i in range(len(self.stack.layers) - 1, -1, -1):
            l = self.stack.layers[i]
            item = QListWidgetItem(l.name)
            item.setIcon(self._make_thumb(l.image, i))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if l.visible else Qt.Unchecked)
            item.setData(Qt.UserRole, i)
            self.list.addItem(item)
        for row in range(self.list.count()):
            if self.list.item(row).data(Qt.UserRole) == self.stack.current:
                self.list.setCurrentRow(row)
                break
        self.list.blockSignals(False)
        self._updating = False
        self._sync_controls()

    def refresh_current_thumb(self):
        """只更新当前图层那一行的缩略图（用于绘制过程中实时反映）"""
        cur = self.stack.current
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.data(Qt.UserRole) == cur:
                l = self.stack.layers[cur]
                item.setIcon(self._make_thumb(l.image, cur))
                break

    def set_current_frame(self, idx):
        self._updating = True
        self.list.blockSignals(True)
        if 0 <= idx < self.list.count():
            self.list.setCurrentRow(idx)
        self.list.blockSignals(False)
        self._updating = False

    def on_row_changed(self, row):
        if self._updating or row < 0:
            return
        if self._multiframe:
            idx = self.list.item(row).data(Qt.UserRole)
            self._main_window.switch_frame(idx)
            return
        idx = self.list.item(row).data(Qt.UserRole)
        if self.stack.current != idx:
            self.mw.canvas.commit_all()
            self.stack.current = idx
            self.mw.canvas.refresh()
        self._sync_controls()

    def on_item_changed(self, item):
        if self._updating:
            return
        # 帧模式下没有"图层可见性"概念，忽略
        if self._multiframe:
            return
        idx = item.data(Qt.UserRole)
        if idx is None or idx < 0 or idx >= len(self.stack.layers):
            return
        self.stack.layers[idx].visible = (item.checkState() == Qt.Checked)
        self.mw.canvas.refresh()

    def on_double_click(self, item):
        if self._multiframe:
            return
        self.rename_layer()

    def add_layer(self):
        self.mw.canvas.commit_all()
        s = self.stack
        s.layers.append(Layer(Image.new("RGBA", (s.width(), s.height()), (0, 0, 0, 0)),
                              f"图层 {len(s.layers)}"))
        s.current = len(s.layers) - 1
        self.refresh_list()
        self.mw.canvas.refresh()
        self.mw.push_history()

    def del_layer(self):
        if len(self.stack.layers) <= 1:
            QMessageBox.warning(self, "提示", "至少保留一个图层")
            return
        self.mw.canvas.commit_all()
        del self.stack.layers[self.stack.current]
        self.stack.current = max(0, self.stack.current - 1)
        self.refresh_list()
        self.mw.canvas.refresh()
        self.mw.push_history()

    def move_up(self):
        self.mw.canvas.commit_all()
        s = self.stack; i = s.current
        if i < len(s.layers) - 1:
            s.layers[i], s.layers[i + 1] = s.layers[i + 1], s.layers[i]
            s.current = i + 1
            self.refresh_list()
            self.mw.canvas.refresh()
            self.mw.push_history()

    def move_down(self):
        self.mw.canvas.commit_all()
        s = self.stack; i = s.current
        if i > 0:
            s.layers[i], s.layers[i - 1] = s.layers[i - 1], s.layers[i]
            s.current = i - 1
            self.refresh_list()
            self.mw.canvas.refresh()
            self.mw.push_history()

    def rename_layer(self):
        i = self.stack.current
        name, ok = QInputDialog.getText(self, "重命名", "图层名：",
                                        text=self.stack.layers[i].name)
        if ok and name:
            self.stack.layers[i].name = name
            self.refresh_list()

    def merge_visible(self):
        self.mw.canvas.commit_all()
        s = self.stack
        s.layers = [Layer(s.flatten(), "合并图层")]
        s.current = 0
        self.refresh_list()
        self.mw.canvas.refresh()

    def _sync_controls(self):
        if not self.stack.layers:
            return
        l = self.stack.cur()
        self._updating = True
        self.opacity_slider.setValue(l.opacity)
        idx = {"normal": 0, "multiply": 1, "screen": 2, "overlay": 3}.get(l.blend, 0)
        self.blend_combo.setCurrentIndex(idx)
        self._updating = False

    def on_opacity_changed(self, v):
        if self._updating:
            return
        self.stack.cur().opacity = v
        self.mw.canvas.refresh()

    def on_blend_changed(self, idx):
        if self._updating:
            return
        mode = ["normal", "multiply", "screen", "overlay"][idx]
        self.stack.cur().blend = mode
        self.mw.canvas.refresh()

    def edit_mask(self):
        """打开蒙版编辑对话框：黑白画笔在 mask 上涂"""
        l = self.stack.cur()
        if l.mask is None:
            l.mask = Image.new("L", (self.stack.width(), self.stack.height()), 255)
        dlg = MaskDialog(self.mw, l.mask)
        if dlg.exec_() == QDialog.Accepted:
            l.mask = dlg.result_mask
            self.mw.canvas.refresh()

    def remove_mask(self):
        self.stack.cur().mask = None
        self.mw.canvas.refresh()

    def merge_selected(self):
        """合并列表中选中的多个图层；默认以当前图层为基准向下合并"""
        rows = sorted([self.list.row(item) for item in self.list.selectedItems()])
        if len(rows) < 2:
            self.merge_down()
            return
        # 把 row 转换为 stack.layers 索引
        indices = [self.list.item(r).data(Qt.UserRole) for r in rows]
        indices.sort()   # 从底到顶
        s = self.stack
        # 目标图层：以当前图层为目标（若当前图层不在选中列表，则取最下面那个）
        if s.current in indices:
            target_idx = s.current
        else:
            target_idx = indices[0]

        # 合并：先把所有选中图层（除目标）合成一张，alpha_composite 到目标
        target = s.layers[target_idx]
        for i in indices:
            if i == target_idx:
                continue
            src = s.layers[i]
            if src.visible:
                target.image.alpha_composite(src.render())

        # 从 layers 里移除除目标外的其他选中
        for i in sorted(indices, reverse=True):
            if i == target_idx:
                continue
            del s.layers[i]
            if i < target_idx:
                target_idx -= 1
        s.current = target_idx
        self.refresh_list()
        self.mw.push_history()
        self.mw.canvas.refresh()

    def merge_down(self):
        """向下合并：当前图层与它下面一个图层合并"""
        s = self.stack
        i = s.current
        if i <= 0:
            return
        lower = s.layers[i - 1]
        upper = s.layers[i]
        if upper.visible:
            lower.image.alpha_composite(upper.render())
        del s.layers[i]
        s.current = i - 1
        self.refresh_list()
        self.mw.push_history()
        self.mw.canvas.refresh()

# ================= 主窗口 =================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("简易图片编辑器")
        self.resize(1300, 850)
        self.foreground = QColor(0, 0, 0)
        self.background = QColor(255, 255, 255)
        self.brush_size = 20
        self.brush_hardness = 80
        self.fill_tolerance = 20
        self.shape_fill = False
        self.current_path = None
        self.modified = False
        self.clipboard = None
        self._multiframe = False
        self._frames = []          # PIL Image 列表
        self._frame_index = 0

        self.history = History()
        self.canvas = Canvas(self)
        self.scroll = CanvasScrollArea(self.canvas)

        # ---------- 标尺 + 画布 的网格布局 ----------
        central = QWidget()
        grid = QGridLayout(central)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.h_ruler = RulerWidget(self.canvas, Qt.Horizontal)
        self.v_ruler = RulerWidget(self.canvas, Qt.Vertical)
        # 左上角空白占位
        self.corner = QWidget(self)
        self.corner.setFixedSize(24, 24)
        self.corner.setStyleSheet("background: #f0f0f0;")
        self.corner.setVisible(False)

        grid.addWidget(self.corner,      0, 0)
        grid.addWidget(self.h_ruler, 0, 1)
        grid.addWidget(self.v_ruler, 1, 0)
        grid.addWidget(self.scroll,  1, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(1, 1)
        self.setCentralWidget(central)
        self.h_ruler.setVisible(False)
        self.v_ruler.setVisible(False)

        # 右侧图层面板（停靠窗口）
        self.layer_dock = QDockWidget("图层", self)
        self.layer_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.layer_dock.setFeatures(QDockWidget.DockWidgetMovable |
                                    QDockWidget.DockWidgetFloatable)
        self.layer_panel = LayerPanel(self, self.canvas.stack)
        self.layer_dock.setWidget(self.layer_panel)
        self.layer_dock.setMinimumWidth(220)
        self.addDockWidget(Qt.RightDockWidgetArea, self.layer_dock)
        
        self.build_toolbar()
        self.build_statusbar()
        self.create_menu()
        self._frame_histories = {}
        self.history.clear(self.canvas.snapshot())

        self._play_timer = QTimer(self)
        self._play_timer.setSingleShot(True)
        self._play_timer.timeout.connect(self._play_next_frame)
        self._playing = False
        self._frame_durations = []    # 每帧毫秒数
        self._frame_loop = 0          # 0 = 无限
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        """拖入窗口时，判断是否包含可接受的文件。"""
        if event.mimeData().hasUrls():
            # 有文件才接受
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """拖动中。"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime = event.mimeData()
        if not mime.hasUrls():
            event.ignore()
            return
        files = []
        for url in mime.urls():
            path = url.toLocalFile()
            if path:
                files.append(path)
        if files:
            # event.pos() 是 MainWindow 局部坐标
            self._on_files_dropped(files, drop_pos=event.pos())
        event.acceptProposedAction()

    def toggle_ruler(self):
        on = self.act_show_ruler.isChecked()
        self.h_ruler.setVisible(on)
        self.v_ruler.setVisible(on)
        self.corner.setVisible(on)
        if on:
            self.scroll._sync_rulers()

    def toggle_grid(self):
        self.canvas.show_grid = self.act_show_grid.isChecked()
        self.canvas.update()

    def grid_settings(self):
        size, ok = QInputDialog.getInt(
            self, "网格设置", "网格间距（像素）：",
            self.canvas.grid_size, 1, 512)
        if ok:
            self.canvas.grid_size = size
            self.canvas.update()

    def toggle_play(self):
        if self._playing:
            self.stop_play()
        else:
            self.start_play()

    def start_play(self):
        if not self._multiframe or len(self._frames) < 2:
            return
        self._playing = True
        if hasattr(self, "act_play"):
            self.act_play.setText("停止播放")
        self._schedule_next_frame()

    def _schedule_next_frame(self):
        if not self._playing:
            return
        durations = getattr(self, "_frame_durations", None) or [100] * len(self._frames)
        if len(durations) != len(self._frames):
            durations = [100] * len(self._frames)
        delay = durations[self._frame_index] if durations else 100
        self._play_timer.start(max(10, delay))

    def _play_next_frame(self):
        if not self._multiframe or not self._playing:
            self.stop_play()
            return
        idx = (self._frame_index + 1) % len(self._frames)
        self.switch_frame(idx)
        self._schedule_next_frame()

    def stop_play(self):
        self._playing = False
        self._play_timer.stop()
        if hasattr(self, "act_play"):
            self.act_play.setText("播放")

    def _set_mode_menu_titles(self, mode):
        if not hasattr(self, "layer_menu"):
            return
        self.layer_menu.setTitle(mode)
        if not hasattr(self, "layer_menu_actions"):
            return

        is_mf = (mode == "帧")          # ← 关键：在这里定义

        if mode == "帧":
            texts = {
                "new": "新建帧",
                "del": "删除当前帧",
                "up": "上一帧",
                "down": "下一帧",
                "merge": "合并所有帧",
            }
            if hasattr(self, "act_merge_down"):
                self.act_merge_down.setEnabled(False)
            if hasattr(self, "act_merge_sel"):
                self.act_merge_sel.setEnabled(False)
        else:
            texts = {
                "new": "新建图层",
                "del": "删除当前图层",
                "up": "上移",
                "down": "下移",
                "merge": "合并可见图层",
            }
            if hasattr(self, "act_merge_down"):
                self.act_merge_down.setEnabled(True)
            if hasattr(self, "act_merge_sel"):
                self.act_merge_sel.setEnabled(True)

        for k, act in self.layer_menu_actions.items():
            if act is not None:
                act.setText(texts[k])
                act.setEnabled(True)

        # 图片参数 / 画布参数：帧模式下禁用
        if hasattr(self, "act_adjust_image"):
            self.act_adjust_image.setEnabled(not is_mf)
        if hasattr(self, "act_adjust_canvas"):
            self.act_adjust_canvas.setEnabled(not is_mf)
        # 播放：仅帧模式下可用
        if hasattr(self, "act_play"):
            self.act_play.setEnabled(is_mf)
        
    def save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", "", "项目文件 (*.myp)")
        if not path:
            return False
        try:
            self.canvas.commit_all()
            save_project(path, self.canvas.stack,
                         current_frame=getattr(self, "_frame_index", 0))
            self.current_path = path
            self.modified = False
            self.update_title()
            self.statusBar().showMessage("项目已保存", 2000)
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存项目失败：\n{e}")
            return False

    def open_project(self):
        self._exit_multiframe()
        path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", "项目文件 (*.myp)")
        if not path:
            return
        try:
            stack, cur_frame = load_project(path)
            self.canvas.set_image(
                Image.new("RGBA", (stack.width(), stack.height()), (0, 0, 0, 0)),
                reset_history=False)
            self.canvas.stack = stack
            self.canvas.refresh()
            self.layer_panel.stack = self.canvas.stack
            self.layer_panel.refresh_list()
            self.history.clear(self.canvas.snapshot())
            self.current_path = path
            self.modified = False
            self.update_title()
            self.statusBar().showMessage("项目已打开", 2000)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开项目失败：\n{e}")

    def paste_from_clipboard(self):
        """从系统剪贴板读取图片（或图片文件路径）并粘贴到画布"""
        cb = QApplication.clipboard()
        mime = cb.mimeData()
        img = None
        # 1) 剪贴板里是图片数据（截图、从浏览器/画图复制等）
        if mime.hasImage():
            qimg = cb.image()
            if not qimg.isNull():
                img = qimage_to_pil(qimg)
        # 2) 剪贴板里是文件路径（从资源管理器复制文件）
        elif mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if not path:
                    continue
                if not path.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".gif", ".bmp",
                         ".tif", ".tiff", ".webp")):
                    continue
                try:
                    img = load_image_first_frame(path)
                    break
                except Exception:
                    continue
        # 3) 都拿不到：交给原来的"内部剪贴板"逻辑
        if img is None:
            self.canvas.paste_clipboard()
            return
        self._drop_image_as_selection(img)
        self.statusBar().showMessage("已粘贴为选区", 1500)

    def _drop_image_as_selection(self, img, center_at=None):
        self.canvas.ensure_canvas_size(img.width, img.height)
        cw = self.canvas.stack.width()
        ch = self.canvas.stack.height()
        if center_at is None:
            cx = max(0, (cw - img.width) // 2)
            cy = max(0, (ch - img.height) // 2)
        else:
            cx = max(0, min(center_at.x() - img.width // 2, cw - img.width))
            cy = max(0, min(center_at.y() - img.height // 2, ch - img.height))
        poly = QPolygon([
            QPoint(cx, cy),
            QPoint(cx + img.width, cy),
            QPoint(cx + img.width, cy + img.height),
            QPoint(cx, cy + img.height)])
        self.canvas.group.add_part(
            poly, QRect(cx, cy, img.width, img.height), img, QPoint(cx, cy))
        self.canvas.refresh()
        self.push_history()

    def _on_files_dropped(self, files, drop_pos=None):
        """
        files: 文件路径列表
        drop_pos: 落点在屏幕上的 QPoint（可选），用于决定粘贴到画布哪个位置
        """
        if not files:
            return
        for path in files:
            if not path:
                continue
            if not path.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".gif", ".bmp",
                     ".tif", ".tiff", ".webp", ".ico")):
                continue
            try:
                img = load_image_first_frame(path)
            except Exception as e:
                print("load image failed:", path, e)
                continue

            # 把落点从 MainWindow 坐标转换到画布坐标
            if drop_pos is None:
                center_at = None
            else:
                # drop_pos 是 MainWindow 局部坐标
                canvas_pos = self.canvas.mapFrom(self, drop_pos)
                if self.canvas.rect().contains(canvas_pos):
                    center_at = self.canvas.widget_to_image(canvas_pos)
                else:
                    center_at = None

            self._drop_image_as_selection(img, center_at=center_at)
            self.statusBar().showMessage(f"已拖入：{path}", 2000)
            return

    def push_history(self):
        if self._multiframe:
            # 先写回当前帧，再存历史
            snapshot_img = self.canvas.stack.flatten(self.canvas.group.parts) \
                if self.canvas.group.parts else self.canvas.stack.flatten()
            self._frames[self._frame_index] = snapshot_img
            self.history.push(self.canvas.snapshot())
            self.modified = True
            self.update_title()
            self.layer_panel.refresh_frame_thumb(self._frame_index, snapshot_img)
        else:
            self.history.push(self.canvas.snapshot())
            self.modified = True
            self.update_title()
            self.layer_panel.refresh_list()

    def undo(self):
        snap = self.history.undo()
        if snap is not None:
            self.canvas.restore(snap)
            self.modified = True
            self.update_title()
        else:
            self.statusBar().showMessage("没有可撤销的操作", 1500)

    def redo(self):
        snap = self.history.redo()
        if snap is not None:
            self.canvas.restore(snap)
            self.modified = True
            self.update_title()
        else:
            self.statusBar().showMessage("没有可重做的操作", 1500)

    def build_toolbar(self):
        tb = QToolBar("工具"); tb.setMovable(False); self.addToolBar(tb)
        self.tool_actions = {}

        def add_tool(name, tool):
            act = QAction(name, self); act.setCheckable(True)
            act.triggered.connect(lambda: self.select_tool(tool, act))
            tb.addAction(act); self.tool_actions[tool] = act
            return act

        for name, tool in [
            ("画笔", "brush"), ("擦除", "eraser"), ("取色", "picker"),
            ("填充", "fill"), ("渐变", "gradient"),("拖动", "pan"), ("椭圆选区", "ellipse_select"),
            ("矩形选区", "rect_select"), ("套索", "lasso"), ("文字", "text")]:
            add_tool(name, tool)

        tb.addWidget(QLabel(" 画笔 "))
        self.brush_combo = QComboBox()
        self.brush_combo.addItems(
            ["圆头", "方形", "软喷枪", "颗粒喷枪", "颗粒笔", "涂抹笔"])
        self.brush_combo.currentIndexChanged.connect(self._on_brush_type)
        tb.addWidget(self.brush_combo)

        tb.addWidget(QLabel(" 不透明度 "))
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(1, 100)
        self.opacity_spin.setValue(100)
        self.opacity_spin.valueChanged.connect(
            lambda v: setattr(self.canvas, "brush_opacity", v))
        tb.addWidget(self.opacity_spin)

        # 线条菜单
        line_btn = QToolButton()
        line_btn.setText("线条")
        line_btn.setPopupMode(QToolButton.InstantPopup)
        line_menu = QMenu(line_btn)
        self.line_style = "straight"   # straight / curve
        for label, val in [("直线", "straight"), ("曲线", "curve")]:
            act = line_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(val == "straight")
            act.triggered.connect(lambda _, v=val: self._set_line_style(v, line_menu))
        line_menu.addSeparator()
        self.line_dash = "solid"       # solid / dash / dot
        dash_menu = line_menu.addMenu("虚线样式")
        for label, val in [("实线", "solid"), ("虚线", "dash"), ("点线", "dot")]:
            act = dash_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(val == "solid")
            act.triggered.connect(lambda _, v=val: self._set_line_dash(v, dash_menu))
        line_menu.addSeparator()
        self.line_arrow = "none"       # none / end / both
        arrow_menu = line_menu.addMenu("箭头")
        for label, val in [("无", "none"), ("终点", "end"), ("两端", "both")]:
            act = arrow_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(val == "none")
            act.triggered.connect(lambda _, v=val: self._set_line_arrow(v, arrow_menu))
        line_btn.setMenu(line_menu)
        line_btn.setCheckable(True)
        tb.addWidget(line_btn)
        self.line_btn = line_btn
        self.tool_actions["line"] = None

        # 形状菜单
        shape_btn = QToolButton()
        shape_btn.setText("形状")
        shape_btn.setPopupMode(QToolButton.InstantPopup)
        shape_menu = QMenu(shape_btn)
        self.shape_kind = "rect"       # rect / ellipse / triangle / hexagon / star / heart
        for label, val in [("矩形", "rect"), ("椭圆", "ellipse"), ("等腰三角形", "triangle"), 
                           ("菱形", "diamond"), ("对称五边形", "pentagon"), ("对称六边形", "hexagon"),
                           ("对称八边形", "octagon"), ("四角弧星", "four_arc_star"), ("五角星", "star"), ("心形", "heart")]:
            act = shape_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(val == "rect")
            act.triggered.connect(lambda _, v=val: self._set_shape_kind(v, shape_menu))
        shape_btn.setMenu(shape_menu)
        shape_btn.setCheckable(True)
        tb.addWidget(shape_btn)
        self.shape_btn = shape_btn
        self.tool_actions["shape"] = None
        self.tool_actions["brush"].setChecked(True)

        tb.addSeparator(); tb.addWidget(QLabel(" 粗细 "))
        sp = QSpinBox(); sp.setRange(1, 300); sp.setValue(self.brush_size)
        sp.valueChanged.connect(lambda v: (setattr(self, "brush_size", v),
                                           self.canvas._rebuild_line_preview()
                                           if self.canvas.line_edit else None))
        tb.addWidget(sp)

        tb.addWidget(QLabel(" 硬度 "))
        hs = QSlider(Qt.Horizontal); hs.setRange(0, 100); hs.setValue(self.brush_hardness)
        hs.setFixedWidth(100)
        hs.valueChanged.connect(lambda v: setattr(self, "brush_hardness", v))
        tb.addWidget(hs)

        tb.addWidget(QLabel(" 容限 "))
        tol = QSpinBox(); tol.setRange(0, 255); tol.setValue(self.fill_tolerance)
        tol.valueChanged.connect(lambda v: setattr(self, "fill_tolerance", v))
        tb.addWidget(tol)

        self.fill_check = QCheckBox("形状填充")
        self.fill_check.stateChanged.connect(
            lambda s: setattr(self, "shape_fill", s == Qt.Checked))
        tb.addWidget(self.fill_check)

        tb.addWidget(QLabel(" 渐变 "))
        self.grad_type = QComboBox()
        self.grad_type.addItems(["线性", "径向"])
        tb.addWidget(self.grad_type)

    def _on_brush_type(self, idx):
        mapping = ["round", "square", "airbrush", "grain_airbrush",
                   "grain", "smudge"]
        self.canvas.brush_type = mapping[idx]

    def build_statusbar(self):
        sb = self.statusBar()
        self.pos_label = QLabel("坐标: -, -"); sb.addWidget(self.pos_label)
        self.fg_label = QLabel(); self.bg_label = QLabel()
        for l in (self.fg_label, self.bg_label):
            l.setFixedSize(24, 24); l.setFrameStyle(QLabel.Box)
        self.update_color_labels()
        for text, fn in [("前景色", self.choose_foreground),
                         ("背景色", self.choose_background),
                         ("交换", self.swap_colors)]:
            b = QPushButton(text); b.clicked.connect(fn); sb.addPermanentWidget(b)
        sb.addPermanentWidget(self.fg_label)
        sb.addPermanentWidget(self.bg_label)

    def create_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("文件")
        fm.addAction("新建", self.new_file, "Ctrl+N")
        fm.addAction("打开", self.open_file, "Ctrl+O")
        fm.addAction("打开多帧…", self.open_multiframe, "Ctrl+Shift+O")
        fm.addAction("保存", self.save_file, "Ctrl+S")
        fm.addAction("另存为", self.save_file_as, "Ctrl+Shift+S")
        fm.addSeparator()
        fm.addAction("打开项目…", self.open_project, "Ctrl+Alt+O")
        fm.addAction("保存项目…", self.save_project_as, "Ctrl+Alt+S")
        fm.addSeparator()
        fm.addAction("关闭文件", self.close_file, "Ctrl+W")
        fm.addAction("退出", self.close, "Ctrl+Q")

        em = mb.addMenu("编辑")
        em.addAction("撤销", self.undo, "Ctrl+Z")
        em.addAction("重做", self.redo, "Ctrl+Y")
        em.addSeparator()
        em.addAction("复制选区", self.canvas.copy_selection, "Ctrl+C")
        em.addAction("剪切选区", self.canvas.cut_selection, "Ctrl+X")
        em.addAction("粘贴", self.paste_from_clipboard, "Ctrl+V")
        em.addSeparator()
        em.addAction("全选", self.canvas.select_all, "Ctrl+A")
        em.addAction("取消选择", self.canvas.deselect, "Ctrl+D")
        em.addAction("反选", self.canvas.invert_selection, "Ctrl+Shift+I")
        em.addSeparator()
        em.addAction("删除选区内容", self.canvas.delete_selection, "Delete")

        vm = mb.addMenu("视图")
        zoom_menu = vm.addMenu("缩放视野")
        for label, z in [("50%", 0.5), ("100%", 1.0),
                         ("150%", 1.5), ("200%", 2.0),
                         ("适应窗口", None)]:
            act = QAction(label, self)
            act.triggered.connect(lambda _, zz=z: self._zoom_action(zz))
            zoom_menu.addAction(act)
        self.act_show_ruler = vm.addAction("显示标尺")
        self.act_show_ruler.setCheckable(True)
        self.act_show_ruler.setChecked(False)
        self.act_show_ruler.triggered.connect(self.toggle_ruler)
        self.act_show_grid = vm.addAction("显示像素网格", self.toggle_grid)
        self.act_show_grid.setCheckable(True)
        self.act_show_grid.setChecked(False)
        vm.addAction("网格设置…", self.grid_settings)

        self.layer_menu_actions = {}
        lm = mb.addMenu("图层")
        self.layer_menu = lm
        lm.addAction("显示/隐藏图层面板", self.toggle_layer_dock, "Ctrl+L")
        lm.addSeparator()
        self.layer_menu_actions["new"] = lm.addAction("新建图层", self.layer_new)
        self.layer_menu_actions["del"] = lm.addAction("删除当前图层", self.layer_del)
        lm.addSeparator()
        self.layer_menu_actions["up"] = lm.addAction("上移", self.layer_up)
        self.layer_menu_actions["down"] = lm.addAction("下移", self.layer_down)
        lm.addSeparator()
        self.act_merge_down = lm.addAction("向下合并", self.layer_merge_down, "Ctrl+E")
        self.act_merge_sel = lm.addAction("合并选择", self.layer_merge_sel)
        self.layer_menu_actions["merge"] = lm.addAction("合并可见图层", self.layer_merge)
        lm.addSeparator()
        self.act_play = lm.addAction("播放", self.toggle_play)

        im = mb.addMenu("图像")
        im.addAction("图片参数…", self.adjust_image)
        im.addAction("画布参数…", self.adjust_canvas)
        tr = im.addMenu("变换")
        tr.addAction("旋转 90° 顺时针", lambda: self._transform("rot90cw"))
        tr.addAction("旋转 90° 逆时针", lambda: self._transform("rot90ccw"))
        tr.addAction("旋转 180°", lambda: self._transform("rot180"))
        tr.addAction("水平镜像", lambda: self._transform("flipH"))
        tr.addAction("垂直镜像", lambda: self._transform("flipV"))
        fx = im.addMenu("特效")
        fx.addAction("锐化…", lambda: self._effect("sharpen"))
        fx.addAction("马赛克…", lambda: self._effect("mosaic"))
        fx.addAction("模糊…", lambda: self._effect("blur"))
        fx.addAction("灰度", lambda: self._effect("gray"))
        
    def toggle_layer_dock(self):
        self.layer_dock.setVisible(not self.layer_dock.isVisible())

    def layer_new(self):
        if self._multiframe:
            self._add_frame()
        else:
            self.layer_panel.add_layer()
    def layer_del(self):
        if self._multiframe:
            self._del_frame()
        else:
            self.layer_panel.del_layer()
    def layer_up(self):
        if self._multiframe:
            self.switch_frame(self._frame_index - 1)
        else:
            self.layer_panel.move_up()
    def layer_down(self):
        if self._multiframe:
            self.switch_frame(self._frame_index + 1)
        else:
            self.layer_panel.move_down()
    def layer_merge(self):
        if self._multiframe:
            self._merge_frames()
        else:
            self.layer_panel.merge_visible()
    def layer_merge_down(self): self.layer_panel.merge_down()
    def layer_merge_sel(self): self.layer_panel.merge_selected()

    def _add_frame(self):
        self.commit_current_frame()
        w, h = self._frames[0].size
        self._frames.insert(self._frame_index + 1,
                            Image.new("RGBA", (w, h), (0, 0, 0, 0)))
        self._frame_index += 1
        self.canvas.stack.layers[0].image = self._frames[self._frame_index].copy()
        self.canvas.refresh()
        self._update_frame_list()
        self.push_history()
        self.statusBar().showMessage("已插入新帧", 1500)

    def _del_frame(self):
        if len(self._frames) <= 1:
            QMessageBox.warning(self, "提示", "至少保留一帧")
            return
        self.commit_current_frame()     # ← 先提交当前帧，避免它被删但历史里缺内容
        del self._frames[self._frame_index]
        if self._frame_index >= len(self._frames):
            self._frame_index = len(self._frames) - 1
        self.canvas.stack.layers[0].image = self._frames[self._frame_index].copy()
        self.canvas.refresh()
        self._update_frame_list()
        self.push_history()
        self.statusBar().showMessage("已删除当前帧", 1500)

    def _merge_frames(self):
        if not self._frames:
            return
        w, h = self._frames[0].size
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        for fr in self._frames:
            out.alpha_composite(fr)
        self._frames = [out]
        self._frame_index = 0
        self.canvas.stack.layers[0].image = out.copy()
        self.canvas.refresh()
        self._update_frame_list()
        self.push_history()
        self.statusBar().showMessage("已合并所有帧", 1500)

    # 工具 / 颜色
    def select_tool(self, tool, action):
        self.canvas._panning = False
        self.canvas._pan_last = None
        if getattr(self.canvas, "line_edit", False) and tool != "line":
            self.canvas._commit_line()
        for t, a in self.tool_actions.items():
            if a is not None:
                a.setChecked(t == tool)
        # 同步两个 QToolButton 的按下态
        if hasattr(self, "line_btn"):
            self.line_btn.setChecked(tool == "line")
            self.shape_btn.setChecked(tool == "shape")
        self.canvas.tool = tool
        self.canvas._update_cursor(self.canvas.mapFromGlobal(QCursor.pos()))

    def set_zoom(self, z):
        self.canvas.zoom = z
        self.canvas.refresh_view()

    def choose_foreground(self):
        c = QColorDialog.getColor(self.foreground, self, "前景色")
        if c.isValid(): self.set_foreground(c)

    def choose_background(self):
        c = QColorDialog.getColor(self.background, self, "背景色")
        if c.isValid(): self.set_background(c)

    def set_foreground(self, c):
        self.foreground = c; self.update_color_labels()
    def set_background(self, c):
        self.background = c; self.update_color_labels()
    def swap_colors(self):
        self.foreground, self.background = self.background, self.foreground
        self.update_color_labels()

    def update_color_labels(self):
        self.fg_label.setStyleSheet(
            f"background-color: {self.foreground.name()}; border: 1px solid #333;")
        self.bg_label.setStyleSheet(
            f"background-color: {self.background.name()}; border: 1px solid #333;")

    def update_pos_label(self, pt, img):
        if 0 <= pt.x() < img.width and 0 <= pt.y() < img.height:
            r, g, b, a = img.getpixel((pt.x(), pt.y()))
            self.pos_label.setText(f"坐标: {pt.x()}, {pt.y()}   颜色: RGBA({r},{g},{b},{a})")
        else:
            self.pos_label.setText("坐标: -, -")

    def update_title(self):
        name = self.current_path if self.current_path else "未命名"
        self.setWindowTitle(f"简易图片编辑器 - {name}{' *' if self.modified else ''}")

    # 文件
    def maybe_save(self):
        if not self.modified:
            return True

        box = QMessageBox(self)
        box.setWindowTitle("未保存的更改")
        box.setText("当前文件有未保存的更改，是否保存？")
        box.setIcon(QMessageBox.Question)
        btn_save = box.addButton("保存", QMessageBox.AcceptRole)
        btn_discard = box.addButton("不保存", QMessageBox.DestructiveRole)
        btn_cancel = box.addButton("取消", QMessageBox.RejectRole)

        box.setDefaultButton(btn_save)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is btn_save:
            return self.save_file() is not False
        if clicked is btn_discard:
            return True
        return False   # 取消

    def new_file(self):
        self._exit_multiframe()
        if not self.maybe_save(): return
        self.canvas.set_image(Image.new("RGBA", (800, 600), (255, 255, 255, 255)), True)
        self.current_path = None; self.modified = False; self.update_title()
        self.layer_panel.stack = self.canvas.stack
        self.layer_panel.refresh_list()

    def open_file(self):
        if not self.maybe_save():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.gif *.bmp *.tif *.tiff *.webp *.ico);;"
            "所有文件 (*)")
        if not path:
            return
        try:
            img = load_image_first_frame(path)   # ← 只取第一帧
            self._exit_multiframe()              # 确保退出帧模式
            self.canvas.set_image(img, reset_history=True)
            self.layer_panel.stack = self.canvas.stack
            self.layer_panel.refresh_list()
            self.current_path = path
            self.modified = False
            self.update_title()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开图片：\n{e}")

    def open_multiframe(self):
        if not self.maybe_save():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "打开多帧图片", "",
            "多帧图片 (*.gif *.tif *.tiff *.webp);;所有文件 (*)")
        if not path:
            return
        try:
            frames = load_image_frames(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开图片：\n{e}")
            return

        if len(frames) < 2:
            QMessageBox.information(
                self, "提示",
                f"该文件只有 {len(frames)} 帧，已按单帧方式打开。")
            self._exit_multiframe()
            self.canvas.set_image(frames[0], reset_history=True)
            self.layer_panel.stack = self.canvas.stack
            self.layer_panel.refresh_list()
        else:
            self._enter_multiframe(frames, path)

        self.current_path = path
        self.modified = False
        self.update_title()

    def _enter_multiframe(self, frames, path):
        self._frame_durations = [100] * len(frames)
        self._frame_loop = 0
        self._multiframe = True
        self._frames = frames
        self._frame_index = 0
        self._frame_histories = {0: History()}
        self._frame_histories[0].clear(self.canvas.snapshot())
        self.history = self._frame_histories[0]
        # 把第一帧当画布内容，同时重置为单图层
        w, h = frames[0].size
        self.canvas.stack = LayerStack(w, h)
        self.canvas.stack.layers = [Layer(frames[0].copy(), "帧 1")]
        self.canvas.stack.current = 0
        self.canvas.group.clear()
        self.canvas.refresh()
        self.layer_panel.set_multiframe(True, self)
        self._update_frame_list()
        # 修改菜单标题
        self._set_mode_menu_titles("帧")
        self.history.clear(self.canvas.snapshot())

    def _exit_multiframe(self):
        if not self._multiframe:
            return
        self.stop_play()
        self._multiframe = False
        self._frames = []
        self._frame_index = 0
        self.layer_panel.set_multiframe(False, self)
        self._set_mode_menu_titles("图层")
        self.history.clear(self.canvas.snapshot())

    def _update_frame_list(self):
        # 让图层面板显示帧
        self.layer_panel.populate_frames(self._frames, self._frame_index)

    def switch_frame(self, idx):
        if not self._multiframe:
            return
        if idx < 0 or idx >= len(self._frames):
            return
        self.commit_current_frame()
        self._frame_index = idx
        self.canvas.stack.layers[0].image = self._frames[idx].copy()
        self.canvas.refresh()
        self.layer_panel.set_current_frame(idx)
        if not self._playing:                # ← 播放中不写历史
            self.history.clear(self.canvas.snapshot())
        #self.push_history()
        self.statusBar().showMessage(
            f"第 {idx + 1} / {len(self._frames)} 帧", 1500)

    def commit_current_frame(self):
        """把画布内容写回当前帧"""
        if not self._multiframe:
            return
        self.canvas.commit_all()
        self._frames[self._frame_index] = self.canvas.stack.flatten()

    def close_file(self):
        self._exit_multiframe()
        if not self.maybe_save(): return
        self.canvas.set_image(Image.new("RGBA", (800, 600), (255, 255, 255, 255)), True)
        self.current_path = None; self.modified = False; self.update_title()
        self.statusBar().showMessage("已关闭文件，新建空白画布", 2000)
        self.layer_panel.stack = self.canvas.stack
        self.layer_panel.refresh_list()

    def save_file(self):
        if self.current_path is None: return self.save_file_as()
        return self._save_to(self.current_path)

    def save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "另存为", "",
                        "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;"
            "GIF (*.gif);;BMP (*.bmp);;ICO (*.ico)")
        if not path: return False
        self.layer_panel.stack = self.canvas.stack
        self.layer_panel.refresh_list()
        return self._save_to(path)

    def _save_to(self, path):
        try:
            self.canvas.commit_all()
            ext = path.lower().rsplit(".", 1)[-1]

            # 多帧模式：保存所有帧
            if self._multiframe:
                self.commit_current_frame()
                frames = self._frames
                durations = getattr(self, "_frame_durations", None) or [100] * len(frames)
                if len(durations) != len(frames):
                    durations = [100] * len(frames)
                loop = getattr(self, "_frame_loop", 0)
                if ext == "gif":
                    frames[0].save(path, "GIF", save_all=True,
                                   append_images=frames[1:],
                                   duration=durations, loop=loop, disposal=2)
                elif ext == "webp":
                    frames[0].save(path, "WEBP", save_all=True,
                                   append_images=frames[1:],
                                   duration=durations, loop=loop, quality=95)
                elif ext in ("tif", "tiff"):
                    frames[0].save(path, "TIFF", save_all=True,
                                   append_images=frames[1:],
                                   duration=durations)
                else:
                    # 其他格式只存第一帧
                    frames[0].save(path)
                self.current_path = path
                self.modified = False
                self.update_title()
                self.statusBar().showMessage("多帧保存成功", 2000)
                return True
            
            img = self.canvas.stack.flatten()
            ext = path.lower().rsplit(".", 1)[-1]
            if ext in ("jpg", "jpeg"):
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                bg.alpha_composite(img)
                bg.convert("RGB").save(path, "JPEG", quality=95)
            elif ext == "png":
                img.save(path, "PNG")
            elif ext == "bmp":
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                bg.alpha_composite(img)
                bg.convert("RGB").save(path, "BMP")
            elif ext == "gif":
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                bg.alpha_composite(img)
                bg.convert("P", palette=Image.ADAPTIVE).save(path, "GIF")
            elif ext == "webp":
                # 支持透明
                img.save(path, "WEBP", quality=95, lossless=False)
            elif ext == "ico":
                # ico 尺寸最大 256，多尺寸保存
                ico_sizes = [(16, 16), (32, 32), (48, 48),
                             (64, 64), (128, 128), (256, 256)]
                # 先合成到白底（ico 不支持半透明看起来更稳）
                img.save(path, "ICO", sizes=ico_sizes)
            else:
                img.save(path)

            self.current_path = path
            self.modified = False
            self.update_title()
            self.statusBar().showMessage("保存成功", 2000)
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：\n{e}")
            return False

    def closeEvent(self, event):
        if self.maybe_save(): event.accept()
        else: event.ignore()

    def _adjust_frame_properties(self):
        if not self._multiframe or not self._frames:
            return
        # 确保 durations 长度匹配
        if len(self._frame_durations) != len(self._frames):
            self._frame_durations = [100] * len(self._frames)
        dlg = FramePropertiesDialog(
            self, self._frames, self._frame_durations, self._frame_loop)
        if dlg.exec_() == QDialog.Accepted:
            self._frame_durations = dlg.result_durations()
            self._frame_loop = dlg.result_loop()
            self.push_history()
            self.statusBar().showMessage("帧属性已更新", 1500)

    def adjust_image(self):
        if self._multiframe:
            self._adjust_frame_properties()
            return
        self.canvas.commit_all()
        dlg = ImageAdjustDialog(self, self.canvas.stack.flatten())
        if dlg.exec_() == QDialog.Accepted:
            self.push_history(); self.canvas.refresh()

    def adjust_canvas(self):
        self.canvas.commit_all()
        dlg = CanvasAdjustDialog(self, self.canvas.stack.flatten())
        if dlg.exec_() == QDialog.Accepted:
            w, h = dlg.result_image.size
            # 对每个图层做同样扩展
            old = self.canvas.stack
            new_stack = LayerStack(w, h, (0, 0, 0, 0))
            new_stack.layers = []
            anchor = dlg.anchor.currentIndex()
            for l in old.layers:
                new = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                if anchor == 0: pos = (0, 0)
                elif anchor == 1: pos = ((w - l.image.width) // 2, (h - l.image.height) // 2)
                else: pos = (w - l.image.width, h - l.image.height)
                new.alpha_composite(l.image, pos)
                new_stack.layers.append(Layer(new, l.name, l.visible))
            if not new_stack.layers:
                new_stack.layers = [Layer(Image.new("RGBA", (w, h), (255, 255, 255, 255)), "背景")]
            new_stack.current = min(old.current, len(new_stack.layers) - 1)
            self.canvas.stack = new_stack
            self.push_history(); self.canvas.refresh()

    def _zoom_action(self, z):
        if z is None:
            area = self.scroll.viewport().size()
            cw, ch = self.canvas.stack.width(), self.canvas.stack.height()
            if cw > 0 and ch > 0:
                z = min(area.width() / cw, area.height() / ch) * 0.98
        self.canvas.zoom = max(0.05, min(4.0, z))
        self.canvas.refresh_view()

    def _transform(self, kind):
        """对选区（如有）或当前图层做几何变换"""
        self.canvas.apply_transform(kind)
        self.push_history()

    def _effect(self, kind):
        self.canvas.apply_effect(kind, self)
        self.push_history()

    def _set_line_style(self, val, menu):
        if getattr(self.canvas, "line_edit", False):
            self.canvas._commit_line()
        self.line_style = val
        for a in menu.actions():
            if a.isCheckable() and a.text() in ("直线", "曲线"):
                a.setChecked(a.text() == ("直线" if val == "straight" else "曲线"))
        # 切到 line 工具，并同步按钮按下态
        self.canvas.tool = "line"
        for t, a in self.tool_actions.items():
            if a is not None:
                a.setChecked(False)
        if hasattr(self, "line_btn"):
            self.line_btn.setChecked(True)
            self.shape_btn.setChecked(False)
        self.canvas._update_cursor(self.canvas.mapFromGlobal(QCursor.pos()))

    def _set_line_dash(self, val, menu):
        self.line_dash = val
        for a in menu.actions():
            a.setChecked(a.text() == {"solid": "实线", "dash": "虚线",
                                       "dot": "点线"}[val])
        if self.canvas.line_edit:
            self.canvas._rebuild_line_preview()
            self.canvas.update()

    def _set_line_arrow(self, val, menu):
        self.line_arrow = val
        for a in menu.actions():
            a.setChecked(a.text() == {"none": "无", "end": "终点",
                                       "both": "两端"}[val])
        if self.canvas.line_edit:
            self.canvas._rebuild_line_preview()
            self.canvas.update()

    def _set_shape_kind(self, val, menu):
        self.shape_kind = val
        mapping = {"rect": "矩形", "ellipse": "椭圆", "triangle": "等腰三角形",
               "hexagon": "对称六边形", "pentagon": "对称五边形","octagon": "对称八边形",
               "diamond": "菱形", "star": "五角星", "four_arc_star": "四角弧星","heart": "心形"}
        for a in menu.actions():
            if a.isCheckable():
                a.setChecked(a.text() == mapping[val])
        self.canvas.tool = "shape"
        for t, a in self.tool_actions.items():
            if a is not None:
                a.setChecked(False)
        if hasattr(self, "shape_btn"):
            self.shape_btn.setChecked(True)
            self.line_btn.setChecked(False)
        self.canvas._update_cursor(self.canvas.mapFromGlobal(QCursor.pos()))

# ================= 图片 / 画布参数 =================

class ImageAdjustDialog(QDialog):
    def __init__(self, parent, pil_img):
        super().__init__(parent)
        self.setWindowTitle("图片参数调整")
        self.img = pil_img.copy(); self.parent_mw = parent
        form = QFormLayout(self)
        self.bright = QSlider(Qt.Horizontal); self.bright.setRange(-100, 100)
        self.contrast = QSlider(Qt.Horizontal); self.contrast.setRange(-100, 100)
        self.sat = QSlider(Qt.Horizontal); self.sat.setRange(-100, 100)
        self.w_spin = QSpinBox(); self.w_spin.setRange(1, 20000); self.w_spin.setValue(self.img.width)
        self.h_spin = QSpinBox(); self.h_spin.setRange(1, 20000); self.h_spin.setValue(self.img.height)
        self.keep = QComboBox(); self.keep.addItems(["保持宽高比", "自由缩放"])
        self.w_spin.valueChanged.connect(self.on_w); self.h_spin.valueChanged.connect(self.on_h)
        self._lock = False
        form.addRow("亮度", self.bright); form.addRow("对比度", self.contrast)
        form.addRow("饱和度", self.sat); form.addRow("宽度", self.w_spin)
        form.addRow("高度", self.h_spin); form.addRow("缩放模式", self.keep)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.apply); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def on_w(self, v):
        if self._lock or self.keep.currentIndex() != 0: return
        self._lock = True
        self.h_spin.setValue(max(1, int(v * self.img.height / self.img.width)))
        self._lock = False

    def on_h(self, v):
        if self._lock or self.keep.currentIndex() != 0: return
        self._lock = True
        self.w_spin.setValue(max(1, int(v * self.img.width / self.img.height)))
        self._lock = False

    def apply(self):
        img = self.img.copy()
        tw, th = self.w_spin.value(), self.h_spin.value()
        if (tw, th) != img.size:
            img = img.resize((tw, th), Image.LANCZOS)
        b = self.bright.value() / 100.0; c = self.contrast.value() / 100.0
        s = self.sat.value() / 100.0
        if b: img = ImageEnhance.Brightness(img).enhance(1 + b)
        if c: img = ImageEnhance.Contrast(img).enhance(1 + c)
        if s: img = ImageEnhance.Color(img).enhance(1 + s)
        self.parent_mw.canvas.set_image(img, reset_history=False)
        self.accept()

class CanvasAdjustDialog(QDialog):
    def __init__(self, parent, pil_img):   
        super().__init__(parent)
        self.setWindowTitle("画布参数调整")
        self.img = pil_img.copy()
        form = QFormLayout(self)
        self.w_spin = QSpinBox(); self.w_spin.setRange(1, 20000); self.w_spin.setValue(self.img.width)
        self.h_spin = QSpinBox(); self.h_spin.setRange(1, 20000); self.h_spin.setValue(self.img.height)
        self.anchor = QComboBox(); self.anchor.addItems(["左上", "居中", "右下"])
        self.fill_color = QColor(255, 255, 255, 255)
        form.addRow("宽度", self.w_spin); form.addRow("高度", self.h_spin)
        form.addRow("原图位置", self.anchor)
        pick = QPushButton("选择填充色"); pick.clicked.connect(self.pick)
        self.fill_label = QLabel(); self.fill_label.setFixedSize(24, 24)
        self.update_fill()
        row = QHBoxLayout(); row.addWidget(pick); row.addWidget(self.fill_label)
        form.addRow("填充色", row)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.apply); btns.rejected.connect(self.reject)
        form.addRow(btns)
        self.result_image = self.img

    def pick(self):
        c = QColorDialog.getColor(self.fill_color, self, "填充色")
        if c.isValid(): self.fill_color = c; self.update_fill()
    def update_fill(self):
        self.fill_label.setStyleSheet(
            f"background-color: {self.fill_color.name()}; border: 1px solid #333;")
    def apply(self):
        w, h = self.w_spin.value(), self.h_spin.value()
        canvas = Image.new("RGBA", (w, h), self.fill_color.getRgb())
        a = self.anchor.currentIndex()
        if a == 0: pos = (0, 0)
        elif a == 1: pos = ((w - self.img.width) // 2, (h - self.img.height) // 2)
        else: pos = (w - self.img.width, h - self.img.height)
        canvas.alpha_composite(self.img, pos)
        self.result_image = canvas
        self.accept()

class Layer:
    def __init__(self, img, name="图层", visible=True,
                 opacity=100, blend="normal", mask=None):
        self.image = img             # PIL RGBA
        self.name = name
        self.visible = visible
        self.opacity = opacity       # 0-100
        self.blend = blend           # normal/multiply/screen/overlay
        self.mask = mask             # PIL "L" 或 None

    def copy(self):
        return Layer(self.image.copy(), self.name, self.visible,
                     self.opacity, self.blend,
                     self.mask.copy() if self.mask else None)

    def render(self):
        """返回应用了蒙版和不透明度的 RGBA 图"""
        img = self.image
        if self.mask is not None:
            # 蒙版应用到 alpha
            r, g, b, a = img.split()
            a = ImageChops.multiply(a, self.mask)
            img = Image.merge("RGBA", (r, g, b, a))
        if self.opacity < 100:
            r, g, b, a = img.split()
            a = a.point(lambda v: int(v * self.opacity / 100))
            img = Image.merge("RGBA", (r, g, b, a))
        return img

class MaskDialog(QDialog):
    MAX_W = 640
    MAX_H = 480

    def __init__(self, parent, mask):
        super().__init__(parent)
        self.setWindowTitle("编辑蒙版")
        self.mask = mask.copy()
        self.brush_size = 30
        self.painting = False
        self.value = 0   # 0=黑(隐藏) 255=白(显示)

        v = QVBoxLayout(self)

        # 用 QWidget 自绘，避免 QLabel 居中留白问题
        self.surface = MaskSurface(self)
        v.addWidget(self.surface)

        row = QHBoxLayout()
        row.addWidget(QLabel("画笔"))
        s = QSlider(Qt.Horizontal); s.setRange(1, 200); s.setValue(self.brush_size)
        s.valueChanged.connect(self._on_size_changed)
        self.size_slider = s
        row.addWidget(s)
        b_black = QPushButton("涂黑(隐藏)")
        b_black.clicked.connect(lambda: setattr(self, "value", 0))
        b_white = QPushButton("涂白(显示)")
        b_white.clicked.connect(lambda: setattr(self, "value", 255))
        row.addWidget(b_black); row.addWidget(b_white)
        v.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

        self.result_mask = self.mask
        self._update_surface_size()

    def _on_size_changed(self, v):
        self.brush_size = v
        self.surface.update()

    def _update_surface_size(self):
        w, h = self.mask.width, self.mask.height
        scale = min(self.MAX_W / w, self.MAX_H / h, 1.0)
        sw, sh = int(w * scale), int(h * scale)
        self.surface.setFixedSize(sw, sh)

    def paint_brush(self, pos):
        """把 widget 坐标转为 mask 坐标，画一个圆"""
        if self.surface.width() == 0 or self.surface.height() == 0:
            return
        sx = self.mask.width / self.surface.width()
        sy = self.mask.height / self.surface.height()
        x = int(pos.x() * sx)
        y = int(pos.y() * sy)
        r = max(1, int(self.brush_size / 2 * sx))
        ImageDraw.Draw(self.mask).ellipse(
            [x - r, y - r, x + r, y + r], fill=self.value)
        self.result_mask = self.mask
        self.surface.update()


class MaskSurface(QWidget):
    """专门用来显示蒙版并接收鼠标涂抹的 widget"""
    def __init__(self, parent_dialog):
        super().__init__()
        self.dlg = parent_dialog
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(120, 120, 120))
        q = pil_to_qimage(self.dlg.mask.convert("RGBA"))
        pm = QPixmap.fromImage(q).scaled(
            self.width(), self.height(),
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(0, 0, pm)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dlg.painting = True
            self.dlg.paint_brush(event.pos())

    def mouseMoveEvent(self, event):
        if self.dlg.painting:
            self.dlg.paint_brush(event.pos())

    def mouseReleaseEvent(self, event):
        self.dlg.painting = False

class FramePropertiesDialog(QDialog):
    def __init__(self, parent, frames, durations, loop):
        super().__init__(parent)
        self.setWindowTitle("帧属性")
        form = QFormLayout(self)

        # 统一帧间隔
        self.uniform_check = QCheckBox("统一帧间隔")
        self.uniform_check.setChecked(True)
        self.uniform_spin = QSpinBox()
        self.uniform_spin.setRange(10, 5000)
        self.uniform_spin.setValue(durations[0] if durations else 100)
        self.uniform_spin.setSuffix(" ms")
        form.addRow(self.uniform_check, self.uniform_spin)

        # 逐帧间隔列表
        self.list = QListWidget()
        for i, d in enumerate(durations):
            item = QListWidgetItem(f"帧 {i + 1}: {d} ms")
            item.setData(Qt.UserRole, d)
            self.list.addItem(item)
        self.list.itemDoubleClicked.connect(self._edit_frame_duration)
        form.addRow("逐帧间隔（双击修改）", self.list)

        # 循环次数
        self.loop_spin = QSpinBox()
        self.loop_spin.setRange(0, 10000)
        self.loop_spin.setValue(loop)
        self.loop_spin.setSpecialValueText("无限循环")
        form.addRow("循环次数（0=无限）", self.loop_spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

        self.uniform_check.stateChanged.connect(self._toggle_uniform)
        self._toggle_uniform()

    def _toggle_uniform(self):
        on = self.uniform_check.isChecked()
        self.uniform_spin.setEnabled(on)
        self.list.setEnabled(not on)

    def _edit_frame_duration(self, item):
        d, ok = QInputDialog.getInt(
            self, "修改帧间隔", "毫秒：", item.data(Qt.UserRole), 10, 5000)
        if ok:
            item.setData(Qt.UserRole, d)
            idx = self.list.row(item)
            item.setText(f"帧 {idx + 1}: {d} ms")

    def result_durations(self):
        if self.uniform_check.isChecked():
            return [self.uniform_spin.value()] * self.list.count()
        return [self.list.item(i).data(Qt.UserRole)
                for i in range(self.list.count())]

    def result_loop(self):
        return self.loop_spin.value()

class RulerWidget(QWidget):
    """通用标尺：水平或垂直。跟随 canvas 的 zoom 和滚动。"""
    def __init__(self, canvas, orientation, dpi=96):
        super().__init__()
        self.canvas = canvas
        self.orientation = orientation    # Qt.Horizontal / Qt.Vertical
        self.dpi = dpi
        self.offset = 0                    # 画布在滚动区里的偏移（像素）
        if orientation == Qt.Horizontal:
            self.setFixedHeight(24)
        else:
            self.setFixedWidth(24)
        self.setMouseTracking(True)

    def set_offset(self, off):
        self.offset = off
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(240, 240, 240))
        p.setPen(QColor(80, 80, 80))
        zoom = self.canvas.zoom
        if zoom <= 0:
            return

        # 选择刻度步长：让相邻主刻度 >= 60px
        step_img = 1
        for s in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
            if s * zoom >= 60:
                step_img = s
                break
        else:
            step_img = 1000

        if self.orientation == Qt.Horizontal:
            img_start = self.offset / zoom
            img_end = (self.offset + self.width()) / zoom
        else:
            img_start = self.offset / zoom
            img_end = (self.offset + self.height()) / zoom

        first = int(img_start // step_img) * step_img
        v = first
        font = p.font(); font.setPointSize(8); p.setFont(font)
        while v <= img_end:
            pos = int(v * zoom - self.offset)
            if self.orientation == Qt.Horizontal:
                p.drawLine(pos, self.height() - 6, pos, self.height())
                p.drawText(pos + 2, 12, str(v))
            else:
                p.drawLine(self.width() - 6, pos, self.width(), pos)
                p.save()
                p.translate(10, pos - 2)
                p.rotate(-90)
                p.drawText(0, 0, str(v))
                p.restore()
            v += step_img
        p.end()

# ================= 入口 =================

def main():
    app = QApplication(sys.argv)
    QApplication.setAttribute(Qt.AA_DisableWindowContextHelpButton, True)

    # 加载 Qt 内置中文翻译
    translator = QTranslator()
    trans_dir = resource_path("translations")
    if translator.load("qt_zh_CN", trans_dir):
        app.installTranslator(translator)
        
    app.setWindowIcon(QIcon(resource_path("app.ico")))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
