import gradio as gr
from PIL import Image
from pinguo1 import search_similar_image

search_size = 4 * 2


def search_process(in_img) -> list:
    if in_img is None:
        return []
    return search_similar_image(in_img, search_size)


def create_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=8):
                in_img = gr.Image(type='pil')
            with gr.Column(scale=2):
                bt_submit = gr.Button('开始搜寻')
        with gr.Row():
            out_img = gr.Gallery().style(columns=[4], rows=[2])

        bt_submit.click(search_process, inputs=[
            in_img
        ], outputs=[
            out_img
        ])
    return demo


if __name__ == '__main__':
    ui = create_ui()
    ui.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
    )