### CARLA environment Codebase for CARLA-Gym

### Usage guide

#### 1. Running scenarios

`python env/envs/intersection/urban_signal_intersection_1b2c1p.py`
##### Config docs:
```bash
{
    "actors": {
            "car1": {
                "type": "vehicle_4W",
                "enable_planner": true,
                "convert_images_to_video": false,
                "early_terminate_on_collision": true,
                "reward_function": "corl2017",
                "scenarios": "SSUIC3_TOWN3_CAR1",
                "manual_control": false,
                "auto_control": false,
                "camera_type": "rgb",
                "collision_sensor": "on",
                "lane_sensor": "on",
                "log_images": false,
                "log_measurements": false,
                "render": true,
                "x_res": 84,  --> Observation dimension along x (height)
                "y_res": 84,  --> Observation dimension along y (width)
                "use_depth_camera": false,
                "send_measurements": false
            }
            }
}
```

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

    Example: Download the 0.9.4 release version from: [Here](https://drive.google.com/open?id=1Wt2cxXCtWI3cSI4rt3_HjGnVfkK8Z9blhttps://github.com/carla-simulator/carla/releases)
    Extract it into `~/software`

    `export CARLA_SERVER=~/software/CARLA_0.9.4/CarlaUE4.sh`
3.1 `pip install carla==0.9.4`
    
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

  
