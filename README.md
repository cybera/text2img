# text2img

This repository is part of the workshop covered under the [Applied Data Science Lab (ADSL)](https://www.cybera.ca/adsl/) program.

### Requirements
1. Ubuntu 20.04
2. CUDA Toolkit 11.6
3. Docker 23.0.1
4. Docker Compose v2.3.3

### Upgrading CUDA Version

The code in the repository may not run for CUDA versions below 11.6. This will prevent the user to take advantage of the GPU capability and will significantly slow down the inference time. 

To upgrade the CUDA version on a Linux machine please follow these general steps:

1. Delete an old NVIDIA installation
2. Download and install the new driver
3. Install the CUDA toolkit

To upgrade CUDA to version 12.2 on **Ubuntu 20.04** OS please follow these steps:

1. Run `$nvidia-smi$` command to check for CUDA version.
2. Delete an old NVIDIA installation with

```
sudo apt-get --purge remove "*nvidia*"
```

3. Find the driver needed on the nvidia drivers page (https://www.nvidia.com/en-us/drivers/unix/). The link "Latest Production Branch Version: 535.129.03" will yield CUDA version 12.2. Place the downloaded file onto your machine.
4. Install the driver and CUDA with: 

```
chmod +x NVIDIA-Linux-x86_64-535.113.01.run
sudo ./NVIDIA-Linux-x86_64-535.113.01.run
```

5. Install the CUDA toolkit by following the steps in 'Installing with Apt' and then the steps in 'Configuring Docker' sections found at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
6. Run `$ nvidia-smi` command to check for the new CUDA version.

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
