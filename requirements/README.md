`requirements` folder contains the files needed to install python dependencies for each describer.

The installation is launched by bvqa `build` command. It creates a separate python envirnment for each describer D, under venvs/D.

For a given describer D, it will:
1. install the base requirements from `base.txt`
2. install the specific requirements from `D.bash`
3. if D.bash does not exist, it will use the default requirements from `D.txt`

Guidelines for maintainers:
- try to pin your dependencies to a specific version whenever possible
- only include dependencies that are necessary for the describer to function properly
- if pytorch is needed, then install it in your B.bash by calling torch.bash
- if you use D.bash, try to list as many dependencies as posssible in your D.txt, and explicitly call pip -r D.txt
- before bvqa executes D.bash the current working directory is the `requirements` folder and the virtual environment has already been activated
- test your installation by running bash test/requirements/all-docker.bash

