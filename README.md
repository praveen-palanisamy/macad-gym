### CARLA-Gym: Multi-Agent Connected Autonomous Driving (MACAD) RL learning platform

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

1. [Getting Started](#getting-started)
2. [Developer Contribution Guide](README.md#developer-contribution-guide)

### Getting Started

0. Assumes an Ubuntu (16.04/18.04 or later) system.
1. Install the system requirements:
	- Anaconda 3.x
		- `wget -P ~ https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash ~/Miniconda3-latest-Linux-x86_64.sh`
	- cmake (`sudo apt-get install cmake`)
	- zlib (`sudo apt-get install zlib1g-dev`)
	- [optional] ffmpeg (`sudo apt-get install ffmpeg`)
	
2. Create a new conda env named "carla_gym" and install the required packages

    `conda env create -f conda_env.yaml`
    
3. Setup CARLA (0.9.x)

    3.1 `mkdir ~/software && cd ~/software`

    3.2 Example: Download the 0.9.4 release version from: [Here](https://drive.google.com/file/d/1p5qdXU4hVS2k5BOYSlEm7v7_ez3Et9bP/view)
    Extract it into `~/software/CARLA_0.9.4`

    3.3 `echo "export CARLA_SERVER=${HOME}/software/CARLA_0.9.4/CarlaUE4.sh" >> ~/.bashrc`
	3.4 `source activate carla_gym` and `pip install carla==0.9.4` (or `easy_install carla*.egg`)
    
4. Clone this repository into your workspace (assuming the path is $workspace on your machine)

    `cd $workspace`

    `git clone <REPO URL>`
    
5. [Work in progress] Test if everything is fine
 
    `cd <REPO DIR>`
    
    `python -m env.carla.multi_env`

### Developer Contribution Guide

- Be sure to `source activate carla_gym` before developing/testing
- Be sure to `git pull` often to make sure you are working on the latest copy
- Follow [PEP8 coding standards](https://www.python.org/dev/peps/pep-0008/) and [Google style](http://google.github.io/styleguide/pyguide.html). Before pushing your local commits, Run `bash .ci/format_code.sh`, this will reformat the files you have changed to automatically make them follow the recommended standard and show you the list of updated file. Review the changes and if all looks good, stage them again and then push
  - You can also auto format specific files using this command: `bash .ci/format_code.sh --files <PATH_TO_FILE1> <PATH_TO_FILE2>`
- Make sure the CI pipeline reports success. If not, please look at the CI run's log file, fix the reported errors

  
