# ─── Model Settings ───────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"       # auto-downloaded on first run
DETECTION_CONF = 0.40            # YOLO confidence threshold (lowered for better recall)
DETECTION_IOU  = 0.45            # NMS IoU threshold

# ─── Hand Tracking ────────────────────────────────────────────────────────────
MAX_HANDS = 1
HAND_DETECT_CONF = 0.65          # slightly relaxed for faster re-detection
HAND_TRACK_CONF  = 0.50

# ─── Guidance Thresholds ──────────────────────────────────────────────────────
# All spatial thresholds are normalised (0–1) relative to frame dimensions.
#
# Tuning guide (640×480 camera, typical desk demo):
#   XY_ALIGN_THRESHOLD  0.12 → ±77 px left/right or ±58 px up/down before
#                              switching to "move forward"
#   CLOSE_DISTANCE_RATIO 0.65 → palm must be within 65% of object's longest
#                              side to trigger grasp instructions
#   GUIDANCE_COOLDOWN_SEC 1.6 → ~1.6 s between repeated TTS phrases
#                              (shorter = more responsive, longer = less noisy)
GUIDANCE_COOLDOWN_SEC = 2.5
XY_ALIGN_THRESHOLD    = 0.12     # ±77 px @ 640 wide  |  ±58 px @ 480 tall
CLOSE_DISTANCE_RATIO  = 0.65     # palm within 65% of max(obj_w, obj_h)

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_ID    = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ─── Chinese → COCO-80 class name mapping ────────────────────────────────────
ZH_TO_COCO: dict[str, str] = {
    "瓶子": "bottle",   "瓶": "bottle",   "水瓶": "bottle",  "饮料瓶": "bottle",
    "杯子": "cup",      "杯": "cup",      "茶杯": "cup",     "水杯": "cup",
    "手机": "cell phone", "电话": "cell phone", "手机手机": "cell phone",
    "苹果": "apple",
    "橙子": "orange",  "橘子": "orange",
    "香蕉": "banana",
    "碗": "bowl",
    "叉子": "fork",
    "勺子": "spoon",   "汤匙": "spoon",
    "刀": "knife",     "餐刀": "knife",
    "书": "book",      "书本": "book",    "书籍": "book",
    "剪刀": "scissors",
    "键盘": "keyboard",
    "鼠标": "mouse",
    "遥控器": "remote", "遥控": "remote",
    "笔记本": "laptop", "电脑": "laptop",  "笔记本电脑": "laptop",
    "时钟": "clock",   "钟": "clock",     "表": "clock",
    "花瓶": "vase",
    "牙刷": "toothbrush",
    "玩具熊": "teddy bear", "泰迪熊": "teddy bear",
    "电视": "tv",      "电视机": "tv",
    "披萨": "pizza",   "蛋糕": "cake",    "甜甜圈": "donut",
    "三明治": "sandwich",
    "胡萝卜": "carrot",
}

# ─── Guidance Messages (Chinese, spoken via TTS) ──────────────────────────────
GUIDANCE_TEXTS: dict[str, str] = {
    "no_target":    "正在搜索目标，请稍候",
    "no_hand":      "请将手放入画面",
    "target_found": "检测到目标，请伸出手",
    "move_right":   "向右移动",
    "move_left":    "向左移动",
    "move_down":    "向下移动",
    "move_up":      "向上移动",
    "move_forward": "向前靠近目标",
    "open_hand":    "张开手",
    "grasp":        "握住目标",
    "success":      "抓取成功",
}

# OpenCV overlay uses English (no CJK font needed)
CV_TEXTS: dict[str, str] = {
    "no_target":    "Searching target...",
    "no_hand":      "Show your hand",
    "move_right":   ">> Move RIGHT",
    "move_left":    "<< Move LEFT",
    "move_down":    "vv Move DOWN",
    "move_up":      "^^ Move UP",
    "move_forward": ">> Move FORWARD",
    "open_hand":    "OPEN hand",
    "grasp":        "GRASP it!",
    "success":      "SUCCESS!",
}
