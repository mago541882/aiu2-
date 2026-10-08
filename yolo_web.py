import gradio as gr
from ultralytics import YOLO

# 1. 加载训练好的模型（注意路径和你截图里的一致）
# 如果你想用官方模型，就写 "yolov8n.pt"；如果你要用刚才训练的，就写绝对路径或相对路径
model = YOLO(r"C:\Users\Jesus\runs\detect\train\weights\best.pt")

def predict_image(img):
    # 进行推理
    results = model(img)
    # 返回画好框的图片
    return results[0].plot()

# 2. 制作 Web 界面
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="上传图片进行目标检测"),
    outputs=gr.Image(label="检测结果"),
    title="我的 YOLO 实时推理 Web 应用",
    description="上传一张图片，本地 YOLO 模型会自动识别图中的物体并画框。"
)

# 3. 启动本地服务器
demo.launch()