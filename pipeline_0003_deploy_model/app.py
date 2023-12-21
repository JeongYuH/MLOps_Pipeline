from __future__ import annotations

import os

import gradio as gr
import torch
from gradio_client import Client
from gradio_client.client import Job

DESCRIPTION = "# Comparing image captioning models"
ORIGINAL_SPACE_INFO = """\
- [GIT-large fine-tuned on COCO](https://huggingface.co/spaces/library-samples/image-captioning-with-git)
- [BLIP-large](https://huggingface.co/spaces/library-samples/image-captioning-with-blip)
- [BLIP-2 OPT 6.7B](https://huggingface.co/spaces/merve/BLIP2-with-transformers)
- [BLIP-2 T5-XXL](https://huggingface.co/spaces/hysts/BLIP2-with-transformers)
- [InstructBLIP](https://huggingface.co/spaces/library-samples/InstructBLIP)
- [Fuyu-8B](https://huggingface.co/spaces/adept/fuyu-8b-demo)
"""

torch.hub.download_url_to_file("http://images.cocodataset.org/val2017/000000039769.jpg", "cats.jpg")
torch.hub.download_url_to_file(
    "https://huggingface.co/datasets/nielsr/textcaps-sample/resolve/main/stop_sign.png", "stop_sign.png"
)
torch.hub.download_url_to_file(
    "https://cdn.openai.com/dall-e-2/demos/text2im/astronaut/horse/photo/0.jpg", "astronaut.jpg"
)


def generate_caption_git(image_path: str, return_job: bool = False) -> str | Job:
    try:
        client = Client("library-samples/image-captioning-with-git")
        fn = client.submit if return_job else client.predict
        return fn(image_path, api_name="/caption")
    except Exception:
        gr.Warning("The GIT-large Space is currently unavailable. Please try again later.")
        return ""


def generate_caption_blip(image_path: str, return_job: bool = False) -> str | Job:
    try:
        client = Client("library-samples/image-captioning-with-blip")
        fn = client.submit if return_job else client.predict
        return fn(image_path, "A picture of", api_name="/caption")
    except Exception:
        gr.Warning("The BLIP-large Space is currently unavailable. Please try again later.")
        return ""


def generate_caption_blip2_opt(image_path: str, return_job: bool = False) -> str | Job:
    try:
        client = Client("merve/BLIP2-with-transformers")
        fn = client.submit if return_job else client.predict
        return fn(
            image_path,
            "Beam search",
            1,  # temperature
            1,  # length penalty
            1.5,  # repetition penalty
            api_name="/caption",
        )
    except Exception:
        gr.Warning("The BLIP2 OPT6.7B Space is currently unavailable. Please try again later.")
        return ""


def generate_caption_blip2_t5xxl(image_path: str, return_job: bool = False) -> str | Job:
    try:
        client = Client("hysts/BLIP2-with-transformers")
        fn = client.submit if return_job else client.predict
        return fn(
            image_path,
            "Beam search",
            1,  # temperature
            1,  # length penalty
            1.5,  # repetition penalty
            50,  # max length
            1,  # min length
            5,  # number of beams
            0.9,  # top p
            api_name="/caption",
        )
    except Exception:
        gr.Warning("The BLIP2 T5-XXL Space is currently unavailable. Please try again later.")
        return ""


def generate_caption_instructblip(image_path: str, return_job: bool = False) -> str | Job:
    try:
        client = Client("library-samples/InstructBLIP")
        fn = client.submit if return_job else client.predict
        return fn(
            image_path,
            "Describe the image.",
            "Beam search",
            5,  # beam size
            256,  # max length
            1,  # min length
            0.9,  # top p
            1.5,  # repetition penalty
            1.0,  # length penalty
            1.0,  # temperature
            api_name="/run",
        )
    except Exception:
        gr.Warning("The InstructBLIP Space is currently unavailable. Please try again later.")
        return ""


def generate_caption_fuyu(image_path: str, return_job: bool = False) -> str | Job:
    try:
        client = Client("adept/fuyu-8b-demo")
        fn = client.submit if return_job else client.predict
        return fn(image_path, "Generate a coco style caption.\n", fn_index=3)
    except Exception:
        gr.Warning("The Fuyu-8B Space is currently unavailable. Please try again later.")
        return ""


def generate_captions(image_path: str) -> tuple[str, str, str, str, str, str]:
    jobs = [
        generate_caption_git(image_path, return_job=True),
        generate_caption_blip(image_path, return_job=True),
        generate_caption_blip2_opt(image_path, return_job=True),
        generate_caption_blip2_t5xxl(image_path, return_job=True),
        generate_caption_instructblip(image_path, return_job=True),
        generate_caption_fuyu(image_path, return_job=True),
    ]
    return tuple(job.result() if job else "" for job in jobs)


with gr.Blocks(css="style.css", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath")
            run_button = gr.Button("Caption")
        with gr.Column():
            out_git = gr.Textbox(label="GIT-large fine-tuned on COCO")
            out_blip = gr.Textbox(label="BLIP-large")
            out_blip2_opt = gr.Textbox(label="BLIP-2 OPT 6.7B")
            out_blip2_t5xxl = gr.Textbox(label="BLIP-2 T5-XXL")
            out_instructblip = gr.Textbox(label="InstructBLIP")
            out_fuyu = gr.Textbox(label="Fuyu-8B")

    outputs = [
        out_git,
        out_blip,
        out_blip2_opt,
        out_blip2_t5xxl,
        out_instructblip,
        out_fuyu,
    ]
    gr.Examples(
        examples=[
            "cats.jpg",
            "stop_sign.png",
            "astronaut.jpg",
        ],
        inputs=input_image,
        outputs=outputs,
        fn=generate_captions,
        cache_examples=os.getenv("CACHE_EXAMPLES") == "1",
    )

    with gr.Accordion(label="The original Spaces can be found here:", open=False):
        gr.Markdown(ORIGINAL_SPACE_INFO)

    run_button.click(
        fn=generate_captions,
        inputs=input_image,
        outputs=outputs,
        api_name="caption",
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name='0.0.0.0', server_port=7860)
