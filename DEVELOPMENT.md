# Vision-Assisted Grasping System (COMP5523) 开发文档

文档目的: 统一项目架构理解、通信协议、关键模块职责与调参方式，方便快速上手与后续演进。
文档日期: 2026-03-10

---

## 1. 项目概述

本项目是一个浏览器到服务端的实时视觉指导系统，用于辅助物体抓取。浏览器采集摄像头画面并以 WebSocket 发送 JPEG 帧，后端进行目标检测与手部识别，输出引导指令与状态信息回传到前端进行可视化与语音播报。

核心能力:
- 实时目标检测 (YOLOv8)
- 手部关键点与手势识别 (MediaPipe Tasks GestureRecognizer)
- 位置关系转引导策略 (状态机)
- 前端可视化覆盖层、语音播报与语音输入

---

## 2. 技术栈

后端:
- FastAPI
- Uvicorn
- OpenCV
- Ultralytics YOLOv8
- MediaPipe Tasks

前端:
- HTML/CSS/JavaScript
- WebSocket 流式通信
- Web Speech API (ASR + TTS)

---

## 3. 目录结构

```
app/
  main.py               FastAPI 入口, WS + REST, 静态托管
  camera_processor.py   帧处理主流程
  detector.py           YOLOv8 封装
  hand_tracker.py       手部识别与手势判定
  guidance.py           引导策略状态机
  config.py             配置与中文映射
static/
  index.html            前端页面
  app.js                前端逻辑 (WS/相机/语音/绘制)
  style.css             前端样式
models/
  gesture_recognizer.task  MediaPipe 模型
  hand_landmarker.task     当前未使用
yolov8n.pt              YOLOv8 模型文件
run.py                  本地启动入口
requirements.txt        依赖列表
```

---

## 4. 快速启动

环境要求:
- Python 3.10+
- Chrome 或 Edge (相机权限与 Web Speech API 支持更好)

步骤:
1. 创建虚拟环境
2. 安装依赖: `pip install -r requirements.txt`
3. 启动服务: `python run.py`
4. 浏览器访问: `http://localhost:8000`
5. 允许摄像头权限
6. 通过语音或快捷按钮设置目标

---

## 5. 架构与数据流

```mermaid
flowchart LR
  A["Browser Camera"] -->|JPEG frames (WS binary)| B["FastAPI /ws"]
  B -->|Frame bytes| C["FrameProcessor"]
  C --> D["YOLOv8 ObjectDetector"]
  C --> E["MediaPipe HandTracker"]
  C --> F["GuidancePolicy"]
  F -->|Guidance text| B
  C -->|State JSON| B
  B -->|WS JSON: state/guidance| G["Frontend UI"]
  G -->|Target set/clear JSON| B
```

关键设计点:
- 处理背压: 处理线程忙时丢弃新帧以保证响应性
- 单线程推理: CV/ML 在单线程执行，避免无序并行导致资源争抢

---

## 6. WebSocket 协议

客户端 -> 服务端
1. 二进制消息: JPEG 帧
2. 文本 JSON 消息:
```json
{"type": "set_target", "target": "bottle"}
{"type": "clear_target"}
```

服务端 -> 客户端
1. 引导消息:
```json
{"type": "guidance", "text": "<中文提示>"}
```

2. 状态消息:
```json
{
  "type": "state",
  "data": {
    "hand_detected": true,
    "target_detected": true,
    "hand_open": false,
    "guidance_state": "aligning",
    "last_guidance": "...",
    "target_class": "bottle",
    "target_display": "水瓶",
    "target_bbox_norm": [x1, y1, x2, y2],
    "hand_center_norm": [x, y],
    "hand_landmarks_norm": [[x, y], ...],
    "direction_norm": [dx, dy]
  }
}
```

3. 目标确认:
```json
{"type": "target_confirmed", "display": "水瓶", "coco_class": "bottle"}
```

4. 目标清空:
```json
{"type": "target_cleared"}
```

坐标均为相对画面尺寸的归一化值 [0, 1]。

---

## 7. REST API

用途: 便于测试或外部系统集成。

- `POST /api/set_target`
  - Body: `{ "target": "bottle" }`
  - Returns: `{ "coco_class": "...", "display": "..." }`

- `POST /api/clear_target`
  - Returns: `{ "status": "cleared" }`

- `GET /api/state`
  - Returns: 最新状态 (与 WS state 一致)

---

## 8. 核心模块说明

`app/main.py`
职责:
- WebSocket 生命周期管理
- 帧处理背压控制
- REST 接口
- 静态资源托管

`app/camera_processor.py`
职责:
- JPEG 解码
- 目标检测
- 手部跟踪与手势识别
- 引导策略调用
- 坐标归一化与状态输出

`app/detector.py`
职责:
- YOLOv8 模型封装
- 目标类别过滤
- 返回最高置信度目标

`app/hand_tracker.py`
职责:
- MediaPipe Tasks 手部模型调用
- 手势类别判定
- 手部开合判定 (含回退启发式)

`app/guidance.py`
职责:
- 引导策略状态机
- 冷却时间控制
- 对齐/靠近/抓取流程

`app/config.py`
职责:
- 模型路径
- 检测阈值
- 引导阈值
- 中文目标映射

---

## 9. 配置与调参

主要配置在 `app/config.py`:
- `DETECTION_CONF`, `DETECTION_IOU`
- `HAND_DETECT_CONF`, `HAND_TRACK_CONF`
- `XY_ALIGN_THRESHOLD`
- `CLOSE_DISTANCE_RATIO`
- `GUIDANCE_COOLDOWN_SEC`

调参建议:
- 召回不足: 适当降低 `DETECTION_CONF`
- 引导过于敏感: 增大 `XY_ALIGN_THRESHOLD`
- 抓取指令过早: 降低 `CLOSE_DISTANCE_RATIO`

---

## 10. 模型与资源

- YOLOv8 模型: `yolov8n.pt`
- MediaPipe 手势模型: `models/gesture_recognizer.task`
- `models/hand_landmarker.task` 当前未接入

离线部署时请提前准备 MediaPipe 模型文件。

---

## 11. 性能与稳定性

- 前端默认 15 FPS 发送，兼顾延迟与识别稳定性
- JPEG 质量 0.7
- 后端处理繁忙时丢帧，避免堆积

低配置设备建议:
- 降低 `CAPTURE_FPS`
- 降低摄像头分辨率
- 替换更轻量 YOLO 模型

---

## 12. 安全与隐私

- 默认不持久化图像，仅在内存中处理
- `localhost` 运行时数据不出本机
- 对外部署需启用 HTTPS 与访问控制

---

## 13. 测试指南

建议的手工测试路径:
1. 启动服务并进入页面
2. 允许摄像头权限
3. WebSocket 状态显示已连接
4. 通过按钮与语音设置目标
5. 移动手部观察引导更新
6. 观察框、关键点与方向箭头是否正确绘制
7. 切换 TTS 功能

---

## 14. 常见问题

相机无法启动:
- 使用 Chrome/Edge
- 检查浏览器权限设置

WebSocket 未连接:
- 确认服务启动于 `http://localhost:8000`
- 查看浏览器控制台错误

模型下载失败:
- 手动放置 `models/gesture_recognizer.task`

控制台中文乱码:
- 确保终端编码为 UTF-8

---

## 15. 扩展方向

- 多目标跟踪与优先级
- 多手识别
- 语音个性化与语言切换
- 模型量化与加速
- 摄像头自动标定

---

## 16. 维护规范

- CV/ML 逻辑集中在 `app/`
- UI 改动集中在 `static/`
- 协议变更同步更新本文档

---

## 17. 代码引用

后端:
- `D:\pyproject\CVP\app\main.py`
- `D:\pyproject\CVP\app\camera_processor.py`
- `D:\pyproject\CVP\app\detector.py`
- `D:\pyproject\CVP\app\hand_tracker.py`
- `D:\pyproject\CVP\app\guidance.py`
- `D:\pyproject\CVP\app\config.py`

前端:
- `D:\pyproject\CVP\static\index.html`
- `D:\pyproject\CVP\static\app.js`
- `D:\pyproject\CVP\static\style.css`

入口与依赖:
- `D:\pyproject\CVP\run.py`
- `D:\pyproject\CVP\requirements.txt`

