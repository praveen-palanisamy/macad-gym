### CARLA environment Codebase for CARLA-Gym

**NOTE**:
> The following instructions are for CARLA 0.9.x.
If you are looking for a stable version that "works", use the master branch.
This branch will soon replace the master branch and will become the mainstream.


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


I changed the conda_env.yaml file to avoid some errors.

  
