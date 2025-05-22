import os
import sys
import time
import torch
import gradio as gr
import trimesh
import yaml
import shutil
from zhipuai import ZhipuAI

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

from Cad_VLM.models.text2cad import Text2CAD
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT
from CadSeqProc.cad_sequence import CADSequence

# 配置常量
OUTPUT_DIR = os.path.abspath("output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局状态管理
class AppState:
    def __init__(self):
        self.original_path = None
        self.base_scale = [1.0, 1.0, 1.0]
        self.current_preview = None
        self.latest_scale = [1.0, 1.0, 1.0]

app_state = AppState()

# 模型加载
def load_model(config_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    
    model = Text2CAD(
        text_config=config["text_encoder"],
        cad_config=cad_config
    ).to(device)
    
    if config["test"]["checkpoint_path"]:
        checkpoint = torch.load(config["test"]["checkpoint_path"], map_location=device)
        pretrained_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
        model.load_state_dict(pretrained_dict, strict=False)
    
    model.eval()
    print("模型加载成功")
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("../Cad_VLM/config/inference_user_input.yaml", device)

# 核心生成函数
def test_model(text):
    try:
        pred_dict = model.test_decode(
            texts=[text] if not isinstance(text, list) else text,
            maxlen=MAX_CAD_SEQUENCE_LENGTH,
            nucleus_prob=0,
            topk_index=1,
            device=device
        )
        
        cad_seq = CADSequence.from_vec(
            pred_dict["cad_vec"][0].cpu().numpy(),
            bit=N_BIT,
            post_processing=True
        )
        return cad_seq.create_mesh().mesh, cad_seq
    except Exception as e:
        print(f"Model error: {str(e)}")
        return None, str(e)

# 文件管理
def clean_files(max_keep=5):
    try:
        files = sorted(os.listdir(OUTPUT_DIR), 
                     key=lambda x: os.path.getctime(os.path.join(OUTPUT_DIR, x)))
        for f in files[:-max_keep]:
            os.remove(os.path.join(OUTPUT_DIR, f))
    except Exception as e:
        print(f"Clean error: {str(e)}")

# 翻译
def translate(text):
    prompt = "你是一名精通中英双语且具有机械工程背景的专业翻译官，专门为CAD设计领域提供术语精准的翻译服务。我将提供中文，你负责将其翻译成英文，并以名词形式返回。只需要返回一个英文名词。"
    client = ZhipuAI(api_key="00244246da7e40b68b4bf6522b72c023.jRs7WuZF9HQ9Y0LG")
    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
    )
    translated = response.choices[0].message.content
    print(translated)
    return translated

# 主生成流程
def generate_model(text, scale_x=1.0, scale_y=1.0, scale_z=1.0):
    global app_state
    try:
        clean_files()
        translated = translate(text)
        
        # 生成基础模型
        mesh, cad = test_model(translated)
        print("生成成功")
        if not mesh:
            return None, None, f"生成失败：{cad}"
        
        # 保存原始模型
        timestamp = int(time.time())
        original_path = os.path.join(OUTPUT_DIR, f"original_{timestamp}.stl")
        mesh.export(original_path)
        app_state.original_path = original_path
        app_state.base_scale = [scale_x, scale_y, scale_z]
        
        # 生成初始缩放版本
        scaled = mesh.apply_scale(app_state.base_scale)
        output_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.stl")
        scaled.export(output_path)
        app_state.current_preview = output_path
        app_state.latest_scale = [1.0, 1.0, 1.0]
        
        print(f"生成成功，原始文件：{original_path}")
        return output_path, output_path, output_path
        
    except Exception as e:
        print(f"Generate error: {str(e)}")
        return None, None, f"系统错误：{str(e)}"

# 实时调整函数
def update_preview(live_x, live_y, live_z):
    try:
        print(f"\n=== 调整触发 ===")
        
        # 加载原始模型
        print(f"🔧 加载原始模型：{app_state.original_path}")
        original = trimesh.load(app_state.original_path)
        
        # 应用实时缩放
        actual_scale = [
            app_state.base_scale[0] * live_x,
            app_state.base_scale[1] * live_y,
            app_state.base_scale[2] * live_z
        ]
        print(f"⚖️ 应用缩放比例：{actual_scale}")
        scaled = original.apply_scale(actual_scale)
        
        # 生成新预览文件
        new_filename = f"preview_live_{int(time.time())}.stl"
        preview_path = os.path.join(OUTPUT_DIR, new_filename)
        scaled.export(preview_path)
        print(f"💾 保存预览文件：{preview_path}")
        
        return preview_path, preview_path
    except Exception as e:
        print(f"实时调整失败: {str(e)}")
        return app_state.original_path  # 返回原始路径保证显示不中断

# Gradio界面
with gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Text2CAD")
    
    with gr.Row():
        input_col = gr.Column()
        output_col = gr.Column()
        
        with input_col:
            text_input = gr.Textbox(label="设计描述", 
                                  placeholder="示例：一个中心带通孔的金属支架...",
                                  max_lines=3)
            
            with gr.Accordion("🔧 初始比例设置", open=False):
                init_scale_x = gr.Slider(0.1, 5.0, 1.0, step=0.1, label="X轴")
                init_scale_y = gr.Slider(0.1, 5.0, 1.0, step=0.1, label="Y轴")
                init_scale_z = gr.Slider(0.1, 5.0, 1.0, step=0.1, label="Z轴")
                
            generate_btn = gr.Button("🚀 生成模型", variant="primary")
            
            with gr.Accordion("🔧 实时调整工具", open=False):
                live_scale_x = gr.Slider(0.1, 5.0, 1.0, step=0.1, label="X轴微调")
                live_scale_y = gr.Slider(0.1, 5.0, 1.0, step=0.1, label="Y轴微调")
                live_scale_z = gr.Slider(0.1, 5.0, 1.0, step=0.1, label="Z轴微调")
                gr.Markdown("**提示**：调整后松开滑块即可查看效果")
            
        with output_col:
            model_view = gr.Model3D(
                label="三维预览",
                height=500,
                zoom_speed=1.5,
                # 强制每次更新都重新加载模型
                every=0.1,
                interactive=True,
                # 添加自定义缓存清除参数
                elem_id="live_preview"
            )
            original = gr.File(label="下载原始模型")
            scaled = gr.File(label="下载缩放后的模型")

    examples = gr.Examples(
        examples=[
            ["一个金属环", 1.0, 1.0, 1.0],
            ["一个五角星", 1.5, 1.5, 0.5],
            ["带四个安装孔的矩形板", 1.0, 0.8, 1.2],
            ["圆柱体中心有通孔的结构", 1.0, 1.0, 2.0]
        ],
        inputs=[text_input, init_scale_x, init_scale_y, init_scale_z],
        label="示例提示",
        fn=lambda x,y,z,w: generate_model(x,y,z,w)[1],
        run_on_click=True
    )

    # 事件绑定
    generate_btn.click(
        fn=generate_model,
        inputs=[text_input, init_scale_x, init_scale_y, init_scale_z],
        outputs=[original, scaled, model_view],
        api_name="generate"
    )
    
    for component in [live_scale_x, live_scale_y, live_scale_z]:
        component.release(
            fn=update_preview,
            inputs=[live_scale_x, live_scale_y, live_scale_z],
            outputs=[scaled, model_view]
        )

if __name__ == "__main__":
    # 允许跨域访问（开发环境需要）
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_CACHE"] = "false"
    
    # 启动时清理旧文件
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 启动应用
    demo.launch(
        server_port=7860,
        share=True,
        debug=True,
        show_api=False
    )