## Modular python requirements

```bash
# On HPC only: load python module, something like
module load python/3.11.6-gcc-13.2.0

# create the environement
python3 -m venv venv
source venv/bin/activate

# On HPC only: start an interactive session from a compute node
srun -p interruptible_gpu --pty bash

# install basic requirements for the vqa script
pip install -r build/requirements.txt

# install requirements for the moondream describer (change moondream to the required describer)
pip install -r build/requirements-moondream.txt

```
