This folder containts bash scripts to test the set up of python virtual environment to run bvqa with different describers.

Content:
- all.bash: create an environment and install dependencies for each describer.
- all-docker.bash: run all.bash in a Docker container (see requirements below).
- describer.bash: create an environment and install dependencies for a specific describer. The first argument is the describer name. Default is moondream.
- bvqa.bash: run bvqa --help, this shall work without any thir party package.
- settings.bash: the settings for the describer.bash

The environments are created under /tmp/venvs/<describer>. The dependencies are installed in a virtual environment using pip.

## Testing with docker

To test with docker, please ensure you have docker installed with support for NVIDIA GPUs:

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

