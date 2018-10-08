### CARLA environment Codebase for CARLA-Gym

**NOTE**: The following instructions are for CARLA 0.8.x. Will get updated once 0.9.x is mature in CARLA upstream

### Getting Started

0. Assumes a Ubuntu 16.04 system.
1. Install the system requirements
	- Anaconda 3.x
	- cmake (sudo apt-get install cmake)
	- zlib (sudo apt-get install zlib1g-dev)
	- [optional] ffmpeg (sudo apt-get install ffmpeg)
	
2. Create a new conda env and install the required packages

    `conda env create -f conda_env.yaml`
    
3. Download stable release version of Carla

    `mkdir ~/software && cd ~/software`

    Download the 0.8.2 release version from: [Here](https://drive.google.com/open?id=1ZtVt1AqdyGxgyTm69nzuwrOYoPUn_Dsm)
    Extract it into `~/software`

    `export CARLA_SERVER=~/software/CARLA_0.8.2/CarlaUE4.sh`
    
4. Clone this repository into your workspace (assuming the path is $workspace on your laptop)

    `cd $workspace`

    `git clone ssh://git@bitbucket.org:carla-gym/carla_gym.git
    
5. Test if everything is fine
 
    `cd CARLA-Gym`

    `python -m env.carla.env`
  
     Should bring up a carla window in test mode