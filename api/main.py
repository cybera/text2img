from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from stable_diffusion import run_stable_diffusion

app = FastAPI()


@app.get("/text2img")
def run_text2img(
    prompt: str,
    *,
    seed: int = 1000,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    image = run_stable_diffusion(
        prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    image.save("text2img_output.png")
    return FileResponse("text2img_output.png")
