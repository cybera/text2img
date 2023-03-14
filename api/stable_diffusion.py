import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

model_id = "CompVis/stable-diffusion-v1-4"
token = os.environ["AUTH_TOKEN"]

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=token
)

pipe = pipe.to("cuda")


def run_stable_diffusion(
    prompt: str,
    *,
    seed: int = 1000,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> Image:
    generator = (
        None if seed is None else torch.Generator(device="cuda").manual_seed(seed)
    )

    image: Image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=num_inference_steps,
    ).images[0]

    return image
