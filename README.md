# text2img

This repository is part of the workshop covered under the [Applied Data Science Lab (ADSL)](https://www.cybera.ca/adsl/) program.

This workshop utilizes the stable diffusion method to generate images from text prompts.

This guide is organized as:

1. Start a Rapid Access Cloud (RAC) GPU instance and setup volume
2. Install Docker
3. Attach volume on a RAC GPU instance to ensure adequate disk space
4. Upgrade CUDA version (optional for g2.* and g3.* instances)
5. Install CUDA toolkit
6. Start the jupyter, fastapi, and streamlit services

### 1. Start a Rapid Access Cloud (RAC) GPU instance

Sign up for a RAC account from the [RAC portal](https://rac-portal.cybera.ca/users/sign_in). Ensure you're accessing RAC from the Edmonton region, then Launch a RAC GPU instance and ensure you are able to use `$ ssh` to access the instance.

> &#x26a0; You can use any GPU flavour to host the `text2img` project, but if opting for **g1.\*** make sure to skip the "Upgrading CUDA Version" step

Attach a volume to your instance, at least 80 GB (ideally 100 GB) using the instructions [here.](https://wiki.cybera.ca/display/RAC/Rapid+Access+Cloud+Guide%3A+Part+1#RapidAccessCloudGuide:Part1-Volumes)

### 2. Install Docker

Paste the following script ([from Docker installation website](https://docs.docker.com/engine/install/ubuntu/)) into an `install_docker.sh` file:

```shell
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

and run the file with `$ sh install_docker.sh`. 

To ensure the `docker` command can be accessed without `sudo`, run the following commands:

```shell
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

### 3. Setup a volume on a RAC

Volumes are setup to ensure additional disk space. RAC instances come with 40 GB of disk space by default, but these images will require more space to build.

If you haven't yet, return to the [setup instructions](#1-start-a-rapid-access-cloud-rac-gpu-instance) to mount an adequate volume to your instance.

Format the volume:

```shell
sudo mkfs.ext4 /dev/sdc
```

List all disks from within the instance with: 

```shell
sudo fdisk -l
```

Look for the disk corresponding to the volume with the assigned amount of space. It will be either `/dev/sdb` or `/dev/sdc`. Remember whether 'b' or 'c' applies here, and use that in place of `<mount_point_name>` below.

Create a mount point for the volume: 

```shell
sudo mkdir /mnt/<mount_point_name>
```

Mount the volume device to the mount point:

```shell
sudo mount /dev/sd<mount_point_name> /mnt/<mount_point_name>
```

Permissions may need to be changed on the new volume, as they are initially set to root: 

```shell
sudo chown ubuntu:ubuntu /mnt/<mount_point_name>
```

In order to make sure Docker data is stored on the new mount, we'll have to create a new file that points to the attached volume. First, using either `chmod` or `sudo`, give yourself permission to create a new file, and create `daemon.json` file in the `/etc/docker` directory and paste the following into it.

```shell
{
    "data-root" : "/mnt/<mount_point_name>"
}
```

You must then restart the docker service with:

```shell
sudo service docker restart
```

### 4. Upgrading CUDA Version (optional)

> &#x26a0; This section can optionally be done for g2.\* and g3.\* instances and will improve performance, but needs to be skipped for g1.\* instances.

Upgrading the CUDA version may enable running newer images on the RAC GPU instance. 

To upgrade the CUDA version on a Linux machine please follow these general steps:

1. Delete an old NVIDIA installation
2. Download and install the new driver
3. Install the CUDA toolkit

To upgrade CUDA to version 12.2 on **Ubuntu 20.04** OS please follow these steps:

1. Run `$ nvidia-smi` command to check for CUDA version (should be 10.1)
2. Delete an old NVIDIA installation with

```shell
sudo apt-get --purge remove "*nvidia*"
```

3. Find the driver needed on the nvidia drivers page (https://www.nvidia.com/en-us/drivers/unix/). The link "Latest Production Branch Version: 535.129.03" will yield CUDA version 12.2. Place the downloaded file onto your machine.
4. Install the driver and CUDA with: 

```shell
chmod +x NVIDIA-Linux-x86_64-535.113.01.run
sudo ./NVIDIA-Linux-x86_64-535.113.01.run
```

### 5. Install CUDA Toolkit

Install the CUDA toolkit by following the steps in 'Installing with Apt' and then the steps in 'Configuring Docker' sections found at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Configure the production repository:

```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Update the packages list from the repository:

```shell
sudo apt-get update
```

Install the NVIDIA Container Toolkit packages:

```shell
sudo apt-get install -y nvidia-container-toolkit
```

Now configure docker by configuring the container runtime by using the nvidia-ctk command:

```shell
sudo nvidia-ctk runtime configure --runtime=docker
```

Restart the Docker daemon:

```shell
sudo systemctl restart docker
```

Run `$ nvidia-smi` command to check for the new CUDA version to ensure it's been upgraded.

### 6. Start the jupyter, fastapi, and streamlit services

Clone the repo into your VM:

```shell
git clone https://github.com/cybera/text2img.git
```

Source the HuggingFace AUTH_TOKEN as an environment variable in your terminal 
```
export AUTH_TOKEN=''
```

or create a `.env` file in the same level as the `docker-compose.yml` file and paste:

```yaml
AUTH_TOKEN='<hugging face token>'
```

into it. If following this approach, remember to `source` to restart your shell, or open a new terminal.

Note: If this is your first time using HuggingFace models, please make sure to go through [the documentation](https://huggingface.co/docs/hub/security-tokens) and generate a user access token with the scope as `read`. 

Then, get the running instance of all the docker services by

```shell
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
4. [Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.](https://arxiv.org/abs/2112.10752)
