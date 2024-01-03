# text2img

This repository is part of the workshop covered under the [Applied Data Science Lab (ADSL)](https://www.cybera.ca/adsl/) program.

This workshop utilizes the stable diffusion method to generate images from text prompts.

This guide is organized as:

1. Start a Rapid Access Cloud (RAC) GPU instance
2. Install Docker
3. Setup a volume on a RAC GPU instance to ensure adequate disk space
4. Upgrade CUDA version (optional)
5. Install CUDA toolkit
6. Start the jupyter, fastapi, and streamlit services

### Start a Rapid Access Cloud (RAC) GPU instance

Sing up for a RAC account from https://rac-portal.cybera.ca/users/sign_in. 

Launch a RAC GPU instance using the instructions [here](https://wiki.cybera.ca/display/RAC/GPU+Computing) (note that `g1.*` instances are far more likely to be availabe when considering scaling the project). It can take >30 minutes for the instance to spawn, so don't worry if it seems to be lagging, keep yourself occupied.

After the instance has spun up, ensure you are able to SSH into the instance. Modify the security rules if necessary.

### Install Docker

Paste the following script (found at https://docs.docker.com/engine/install/ubuntu/) into an `install_docker.sh` file:

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

### Setup a volume on a RAC

Volumes are setup to ensure additional disk space. RAC instances come with 40 GB of disk space by default.

In the RAC console, under 'Volumes', create a volume wuth 80-100 GBs of space and attach it to the GPU instance.

Format the volume:

```shell
sudo mkfs.ext4 /dev/sdc
```

List all disks from within the instance with: 

```shell
sudo fdisk -l
```

Look for the disk corresponding to the volume with the assigned amount of space. It will be either `/dev/sdb` or `/dev/sdc`. Remember whether 'b' or 'c' applies here.

Create a mount point for the volume: 

```shell
sudo mkdir /mnt/<mount_point_name>
```

Mount the volume device to the mount point (replace `sdb` with `sdc` as needed):

```shell
sudo mount /dev/<mount_point_name> /mnt/<mount_point_name>
```

Permissions may need to be changed on the new volume, as they are initially set to root: 

```shell
sudo chown ubuntu:ubuntu /mnt/<mount_point_name>
```

In order to make sure docker data is stored on the new mount, create a `daemon.json` file in the `/etc/docker` directory and paste the following into it:

```shell
{
    "data-root" : "/mnt/<mount_point_name>"
}
```

You must then restart the docker service with:

```shell
sudo service docker restart
```

### Specifying CUDA Version

The code in the repository is setup to run for CUDA versions below 11.6.

Upgrading the CUDA version may enable running newer images on the RAC GPU instance. 

To upgrade the CUDA version on a Linux machine please follow these general steps:

1. Delete an old NVIDIA installation
2. Download and install the new driver
3. Install the CUDA toolkit

To upgrade CUDA to version 12.2 on **Ubuntu 20.04** OS please follow these steps:

1. Run `$ nvidia-smi` command to check for CUDA version.
2. Delete an old NVIDIA installation with

```shell
sudo apt-get --purge remove "*nvidia*"
```
Next, install the NVIDIA toolkit by following the steps in 'Installing with Apt' and then the steps in 'Configuring Docker' sections found at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

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

To clean up any NVIDIA modules in memory, use the following steps:
```shell
sudo systemctl isolate multi-user.target
sudo modprobe -r nvidia-drm
```

3. Find the driver needed on the nvidia drivers page (https://www.nvidia.com/en-us/drivers/unix/). Each flavour of GPU on RAC requires different drivers:

| RAC Flavour | GPU | NVIDIA Driver
-|-|-
g1.* | Tesla K80 | 440.95.01
g2.* | Titan XP | 535.146.02
g3.* | Titan RTX | 535.146.02

Download the appropriate driver to your machine:

- [440.95.01](https://www.nvidia.com/Download/driverResults.aspx/160640/en-us/)
- [535.146.02](https://www.nvidia.com/Download/driverResults.aspx/216728/en-us/)

4. Install the driver and CUDA with: 

```shell
chmod +x NVIDIA-Linux-x86_64-###.##.##.run
sudo ./NVIDIA-Linux-x86_64-###.###.##.run
```

(Answer yes to all prompts during installation)

Restart the Docker daemon:

```shell
sudo systemctl restart docker
```

Run `$ nvidia-smi` command to check for the new CUDA version.

### Start the jupyter, fastapi, and streamlit services

First, please source the HuggingFace AUTH_TOKEN as an environment variable in your terminal 
```
export AUTH_TOKEN=''
```

or create a `.env` file in the same level as the `docker-compose.yml` file and paste:

```yaml
AUTH_TOKEN='<hugging face token>'
```

into it.

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
4. Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
