# Vision-Assisted Grasping System (COMP5523)

面向视觉辅助抓取的实时系统。浏览器采集摄像头画面并通过 WebSocket 发送 JPEG 帧，后端执行目标检测与手部识别，返回引导指令与状态数据，前端进行可视化与语音播报。

## 功能概览

- 目标检测: YOLOv8
- 手部关键点与手势识别: MediaPipe Tasks
- 引导策略: 状态机 + 位置关系
- 前端可视化覆盖层 + 语音播报 (TTS) + 语音输入 (ASR)

## 快速开始

环境要求:
- Python 3.10+
- Chrome 或 Edge (相机与 Web Speech API 支持更好)

运行:
```bash
pip install -r requirements.txt
python run.py
```

打开浏览器:
```
http://localhost:8000
```

## 目录结构

```
app/                 后端核心逻辑
static/              前端页面与脚本
models/              模型文件 (含自动下载模型)
run.py               启动入口
requirements.txt     依赖
```

## 通信协议 (摘要)

WebSocket:
- 客户端 -> 服务端: 二进制 JPEG 帧
- 客户端 -> 服务端: JSON 指令
  - {"type":"set_target","target":"bottle"}
  - {"type":"clear_target"}
- 服务端 -> 客户端: JSON 状态与引导

REST:
- POST /api/set_target
- POST /api/clear_target
- GET /api/state

## 常见问题

- 摄像头权限: 使用 Chrome/Edge 并检查权限设置
- WebSocket 无连接: 确认服务运行于 http://localhost:8000
- 模型下载失败: 手动放置 models/gesture_recognizer.task

## 文档

开发文档见:
- `DEVELOPMENT.md`

## 许可

课程项目用途。如需开源许可请补充 LICENSE。
