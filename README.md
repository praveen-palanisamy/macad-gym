### CARLA environment Codebase for CARLA-Gym

**NOTE**:
> The following instructions are for CARLA 0.9.x.
If you are looking for a stable version that "works", use the master branch.
This branch will soon replace the master branch and will become the mainstream.

1. [Getting Started](#getting-started)
2. [Developer Contribution Guide](http://bitbucket.org:carla-gym/carla_gym.git

### Getting Started

0. Assumes a Ubuntu 16.04 system.
1. Install the system requirements
	- Anaconda 3.x
	- cmake (sudo apt-get install cmake)
	- zlib (sudo apt-get install zlib1g-dev)
	- [optional] ffmpeg (sudo apt-get install ffmpeg)
	
2. Create a new conda env and install the required packages

    `conda env create -f conda_env.yaml`
    
3. Download 0.9.x release version of Carla

    `mkdir ~/software && cd ~/software`

    Example: Download the 0.9.0 release version from: [Here](https://drive.google.com/open?id=1JprRbFf6UlvpqX98hQiUG9U4W_E-keiv)
    Extract it into `~/software`

    `export CARLA_SERVER=~/software/CARLA_0.9.0/CarlaUE4.sh`
    
4. Clone this repository into your workspace (assuming the path is $workspace on your laptop)

    `cd $workspace`

    `git clone ssh://git@bitbucket.org:carla-gym/carla_gym.git
    
5. [Work in progress] Test if everything is fine
 
    `cd CARLA-Gym`
    
    `python -m env.carla.multi_env`

### Developer Contribution Guide

- Be sure to `source activate carla_gym` before developing/testing
- Be sure to `git pull` often to make sure you are working on the latest copy
- Follow [PEP8 coding standards](https://www.python.org/dev/peps/pep-0008/) and [Google style](http://google.github.io/styleguide/pyguide.html). Before pushing your local commits, Run `bash .ci/format_code.sh`, this will reformat the files you have changed to automatically make them follow the recommended standard and show you the list of updated file. Review the changes and if all looks good, stage them again and then push
  - You can also auto format specific files using this command: `bash .ci/format_code.sh --files <PATH_TO_FILE1> <PATH_TO_FILE2>`
- Make sure the CI pipeline reports success. If not, please look at the CI run's log file, fix the reported errors

  
