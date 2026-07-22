# 🦿 Hand Gesture-Based Quadruped Robot Control

An intelligent robot control system that enables **real-time control of a quadruped robot using hand gestures**. The system leverages computer vision to recognize hand gestures from a webcam and employs an AI-powered reasoning agent to translate user intentions into robot motion commands. The solution supports both **Gazebo simulation** and deployment on a physical robot through **ROS2**.

## 🚀 Features

* 🎥 Real-time hand gesture recognition using a standard webcam
* 🤖 AI reasoning with a ReAct agent for intent understanding and action planning
* 🦿 Control quadruped robot locomotion and behaviors through intuitive gestures
* 🌐 FastAPI backend for communication between perception, reasoning, and robot control
* 🖥️ User-friendly GUI for monitoring camera input, recognized gestures, and robot status
* ⚡ Low-latency command pipeline for responsive robot control
* 🧪 Support for both Gazebo simulation and real robot deployment
* 🔌 Modular architecture for adding new gestures and robot behaviors

## 🏗️ System Architecture

```
Webcam
   │
   ▼
Hand Gesture Recognition
   │
   ▼
Gesture Classification
   │
   ▼
ReAct Agent
(Intent Reasoning & Action Planning)
   │
   ▼
FastAPI Server
   │
   ▼
ROS2 Interface
   │
   ├────────► Gazebo Simulation
   │
   └────────► Physical Quadruped Robot
```

## 🛠️ Technology Stack

| Component            | Technology |
| -------------------- | ---------- |
| Backend API          | FastAPI    |
| Programming Language | Python     |
| Robot Middleware     | ROS2       |
| Robot Simulation     | Gazebo     |
| Computer Vision      | OpenCV     |
| AI Agent             | ReAct      |
| User Interface       | Python GUI |
| Camera               | USB Webcam |

## 📂 Project Objectives

This project aims to develop an intuitive human–robot interaction system where users can control a quadruped robot naturally through hand gestures. Instead of relying on traditional remote controllers, the robot interprets visual gestures captured by a webcam and converts them into motion commands.

The ReAct (Reasoning and Acting) framework enhances decision-making by combining perception with reasoning, enabling more robust interpretation of user intentions and safer robot behavior.

## Preresiquite
### Install and run docker on MacOS
1. Install Docker Desktop and run
To check docker version,
docker --version

To verify docker is running,
docker run hello-world

2. Create ROS2 Project
mkdir ros2-workspace
cd ros2-workspace
mkdir src
mkdir docker

3. Create Docker file
- Create Dockerfile in folder docker (without .something).
- In the project root (folder ros2-workspace), create docker-compose.yml.

4. Build the Image
In the project root, type docker compose build on CLI.

5. Start the Container
docker compose up -d
Verify that docker is running 
docker ps

6. Enter the Container 
docker exec -it ros2_jazzy bash
Inside the container, source ROS2:
source /opt/ros/jazzy/setup.bash
To verify, type
ros2
The ROS2 command-line help shows.

7. Create a work space
Inside the container,
mkdir -p src
colcon build
source install/setup.bash

8. Test ROS2
Open Terminal, 
docker exec -it ros2_jazzy bash
source /opt/ros/jazzy/setup.bash

ros2 run demo_nodes_cpp talker

Open another Terminal.
docker exec -it ros2_jazzy bash
source /opt/ros/jazzy/setup.bash
ros2 run demo_nodes_cpp listener

Testing ROS2 finishes here.

### Install Gazebo in Docker container
After installing Gazebo in Step 1, you can launch Gazebo Sim, a 3D robotics simulator, from a terminal.

1. aunch Gazebo by running:

gz sim shapes.sdf  # Fortress uses "ign gazebo" instead of "gz sim"
This command will launch both the Sim server and Sim GUI with a world that contains six simple shapes.

Add the -v 4 command line argument to generate error, warning, informational, and debugging messages on the console.

gz sim shapes.sdf -v 4  # Fortress uses "ign gazebo" instead of "gz sim"
Gazebo Sim can also be run headless, i.e. without the GUI, by using the -s (server only) flag.

gz sim -s shapes.sdf -v 4  # Fortress uses "ign gazebo" instead of "gz sim"

2. Configure package repositories for Gazebo Sim

sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update

sudo apt-get install libgz-sim<#>-dev


### Create a robot description package


## 🎯 Example Robot Commands

| Hand Gesture | Robot Action |
| ------------ | ------------ |
| Open Palm    | Move Forward |
| Fist         | Stop         |
| Point Left   | Turn Left    |
| Point Right  | Turn Right   |
| Thumbs Up    | Stand Up     |
| Thumbs Down  | Sit Down     |

## 📌 Applications

* Human–Robot Interaction (HRI)
* Service Robotics
* Educational Robotics
* Search and Rescue Robot Interfaces
* Smart Manufacturing
* AI Robotics Research

## 🔮 Future Improvements

* Voice and gesture multimodal control
* Custom gesture training
* Reinforcement Learning for adaptive behaviors
* Large Vision-Language Model integration
* Multi-camera gesture recognition
* ROS2 Navigation integration
* Real-world deployment on Unitree, Go2, or other quadruped robots

---

**Keywords:** Hand Gesture Recognition, Quadruped Robot, ROS2, Gazebo, FastAPI, Python, Computer Vision, Human-Robot Interaction (HRI), ReAct Agent, Robotics, AI, Real-Time Control.
