"""
第15阶段：Web 部署 - Flask API

本脚本介绍项目如何把训练好的模型部署为 Web 服务，
使用 Flask 提供 REST API 与 WebSocket 实时推理，前端即可调用。

对应文档：docs/usage/04-training.md
"""

import sys
import os


def section_1_why_web():
    """15.1 为什么需要 Web 部署"""
    print("\n" + "=" * 50)
    print("15.1 为什么需要 Web 部署")
    print("=" * 50)
    print("""
前两个阶段你已能在终端里做实时推理（14 阶段）。
但真实系统往往需要：
- 给非技术用户一个网页界面
- 让前端（浏览器）调用识别能力
- 把采集、训练、评估也统一管理起来

本项目用 Flask 提供这套 Web 服务，代码在 api/app.py。
""")


def section_2_endpoints():
    """15.2 主要 API 端点"""
    print("\n" + "=" * 50)
    print("15.2 主要 API 端点")
    print("=" * 50)
    print("""
api/app.py 暴露的端点：

  GET  /              首页
  GET  /dashboard     控制台
  GET  /demo          实时演示页
  GET  /docs          文档查看器

  GET  /api/health        健康检查（是否加载了模型、类别数等）
  POST /api/load_model    加载模型
  POST /api/predict       单帧预测（传 base64 图片或 171 维特征）
  POST /api/detect        仅检测关键点（不预测）
  POST /api/models/load   按文件名加载模型
  GET  /api/models        已训练模型列表
  POST /api/collect       启动数据采集
  POST /api/preprocess    启动预处理
  POST /api/train         启动模型训练
  POST /api/evaluate      启动模型评估
  WS   /ws/detect         WebSocket 实时检测+预测
""")


def section_3_predict_flow():
    """15.3 /api/predict 数据流"""
    print("\n" + "=" * 50)
    print("15.3 /api/predict 数据流")
    print("=" * 50)
    print("""
前端把一帧画面发来，服务器做 检测 → 提取 → 预测：

  客户端 POST /api/predict  {image: "<base64>"}
        │
        ▼
  api_predict()
    │  1. base64 解码 → cv2.imdecode 得到图片
    │  2. extract_landmarks_from_frame():
    │       翻转 → HolisticDetector.detect → get_landmarks
    │       归一化 x,y 到 [0,1] → 得到 171 维 landmarks
    │  3. detect_and_predict():
    │       传统ML: 特征 → scaler → model.predict
    │       深度学习: 累积序列 → 满 30 帧后 model 预测
        │
        ▼
  返回 JSON  {prediction, confidence, has_hand, buffer_size, ...}
""")


def section_4_launch():
    """15.4 启动服务"""
    print("\n" + "=" * 50)
    print("15.4 启动服务")
    print("=" * 50)
    print("""
在项目根目录启动 API 服务：

  conda activate lingxin-gpu
  python api/app.py --model lstm

默认监听 http://localhost:5000，可访问：
  http://localhost:5000/demo          实时识别演示
  http://localhost:5000/dashboard     控制台
  http://localhost:5000/api/health    健康检查

可选参数：
  --model svm|lstm|...  启动时加载的模型（默认 lstm）
  --port 5000           端口
  --host 0.0.0.0        监听地址
""")


def section_5_health_check():
    """15.5 用 requests 测试健康检查"""
    print("\n" + "=" * 50)
    print("15.5 用 requests 测试健康检查")
    print("=" * 50)

    print("""
服务启动后，可验证是否正常：

  import requests
  r = requests.get("http://localhost:5000/api/health")
  print(r.json())
  # 预期输出示例（取决于已加载模型）：
  # {'status': 'ok', 'model_loaded': True, 'model_type': 'lstm',
  #  'num_classes': 10, 'has_scaler': True, 'has_detector': True}
""")

    try:
        import requests
        print("[提示] requests 库已安装，可按照上面的代码测试。")
    except ImportError:
        print("[提示] requests 库未安装，运行 `pip install requests` 后测试。")


def main():
    print("=" * 60)
    print("第15阶段：Web 部署 - Flask API")
    print("=" * 60)

    section_1_why_web()
    section_2_endpoints()
    section_3_predict_flow()
    section_4_launch()
    section_5_health_check()

    print("\n" + "=" * 60)
    print("Web 部署学习完成！")
    print("至此，你已完成从环境准备到实时识别、Web 部署的完整学习路线。")
    print("=" * 60)


if __name__ == '__main__':
    main()
