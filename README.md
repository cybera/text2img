# text2img

This repository is part of the workshop covered under the [Applied Data Science Lab (ADSL)](https://www.cybera.ca/adsl/) program.

### Requirements
1. Ubuntu 20.04
2. CUDA Toolkit 11.6
3. Docker 23.0.1
4. Docker Compose v2.3.3

### Setup
First, please source the HuggingFace AUTH_TOKEN as an environment variable in your terminal 
```
export AUTH_TOKEN=''
```
Note: If this is your first time using HuggingFace models, please make sure to go through [the documentation](https://huggingface.co/docs/hub/security-tokens) and generate a user access token with the scope as `read`. 

Then, get the running instance of all the docker services by

```
cd text2img
docker compose build --parallel
docker compose up
```

After the succesful build, we can access the running services using the following links:

|Service |URL|
|-----|--------|
|JupyterLab|http://localhost:8888/|
|FAST API  |http://localhost:8000/docs|
|Streamlit  |http://localhost:8501/app|

1. If you are running this on a remote cloud server, make sure to do relevant port forwarding

2. The default password for accessing the running instance for JupyterLab container is `gpu-jupyter`

### References
1. [Set up Your own GPU-based Jupyter easily using Docker](https://cschranz.medium.com/set-up-your-own-gpu-based-jupyterlab-e0d45fcacf43)
2. [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion)
3. [DeepLearning AI: FastAPI for Machine Learning: Live coding an ML web application](https://www.youtube.com/watch?v=_BZGtifh_gw)
4. Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
