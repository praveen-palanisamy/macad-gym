### MACAD-Gym

Multi-Agent Connected Autonomous Driving (MACAD) learning platform using CARLA Autonomous Driving simulator.

### Usage guide

1. [Getting Started](#getting-started)
1. [Installation](#installation)
1. [Developer Contribution Guide](CONTRIBUTING.md)
1. [Notes for Gym wrappers for CARLA 0.8.x (stable) versions](README.md#notes)


### Getting Started

> Assumes an Ubuntu (16.04/18.04 or later) system.

1. Install the system requirements:
	- Miniconda/Anaconda 3.x
		- `wget -P ~ https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash ~/Miniconda3-latest-Linux-x86_64.sh`
	- cmake (`sudo apt install cmake`)
	- zlib (`sudo apt install zlib1g-dev`)
	- [optional] ffmpeg (`sudo apt install ffmpeg`)
    
3. Setup CARLA (0.9.x)

    3.1 `mkdir ~/software && cd ~/software`

    3.2 Example: Download the 0.9.4 release version from: [Here](https://drive.google.com/file/d/1p5qdXU4hVS2k5BOYSlEm7v7_ez3Et9bP/view)
    Extract it into `~/software/CARLA_0.9.4`
    
    3.3 `echo "export CARLA_SERVER=${HOME}/software/CARLA_0.9.4/CarlaUE4.sh" >> ~/.bashrc`
    
	
### Installation

 - Option1 for users: `pip install git+https://github.com/praveen-palanisamy/macad-gym.git`
 - Option2 for developers:
     - Fork/Clone the repository to your workspace:
    `git clone https://github.com/praveen-palanisamy/macad-gym.git`
    `cd macad-gym`
     - Create a new conda env named "macad-gym" and install the required packages:
      `conda env create -f conda_env.yaml`
     - Activate the `macad-gym` conda python env:
      `source activate carla-gym`
     - Install CARLA PythonAPI: `pip install carla==0.9.4`
     > NOTE: Change the carla client PyPI package version number to match with your CARLA server version
       

#### Learning Platform and Agent Interface

The MACAD-Gym platform provides learning environments for training agents in 
single-agent and multi-agent settings for various autonomous driving tasks and 
scenarios.
The learning environments following a naming convention for the ID to be consistent
and to support versioned benchmarking of agent algorithms.
The number of training environments in MACAD-Gym is expected to grow over time
(PRs are very welcome!). 
The Environment-Agent interface is fully compatible with the OpenAI-Gym interface
thus, allowing for easy experimentation with existing RL agent algorithm 
implementations and libraries.

##### Environments
To get a list of available environments, you can use
the `get_available_envs()` function as shown in the code snippet below:

```python
import gym
import macad_gym
macad_gym.get_available_envs()
``` 
This will print the available environments. Sample output is provided below for reference:

```bash
Environment-ID: Short description
{'HeteNcomIndePOIntrxMATLS1B2C1PTWN3-v0': 'Heterogeneous, Non-communicating, '
                                          'Independent,Partially-Observable '
                                          'Intersection Multi-Agent scenario '
                                          'with Traffic-Light Signal, 1-Bike, '
                                          '2-Car,1-Pedestrian in Town3, '
                                          'version 0',
 'HomoNcomIndePOIntrxMASS3CTWN3-v0': 'Homogenous, Non-communicating, '
                                     'Independed, Partially-Observable '
                                     'Intersection Multi-Agent scenario with '
                                     'Stop-Sign, 3 Cars in Town3, version 0'}
```

The environment interface is simple and follows the widely adopted OpenAI-Gym
interface. You can create an instance of a learning environment using the 
following 3 lines of code:

```python
import gym
import macad_gym
env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")
```
Like any OpenAI-Gym environment, you can obtain the observation space and action
space using the following attributes:
```bash
>>> print(env.observation_space)
Dict(car1:Box(168, 168, 3), car2:Box(168, 168, 3), car3:Box(168, 168, 3))
>>> print(env.action_space)
Dict(car1:Discrete(9), car2:Discrete(9), car3:Discrete(9))
```

### **NOTEs**:
> MACAD-Gym is for CARLA 0.9.x . If you are
looking for an OpenAI Gym-compatible agent learning environment for CARLA 0.8.x (stable release),
use [this carla_gym environment](https://github.com/PacktPublishing/Hands-On-Intelligent-Agents-with-OpenAI-Gym/tree/master/ch8/environment).
