FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy

# -----------------------------------------------------------------------------
# Development tools
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    nano \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-argcomplete \
    sudo \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Initialize rosdep
# -----------------------------------------------------------------------------
RUN rosdep init || true
RUN rosdep update || true

# -----------------------------------------------------------------------------
# ROS2 desktop packages
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ros-jazzy-desktop \
    ros-jazzy-demo-nodes-cpp \
    ros-jazzy-demo-nodes-py \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# RViz
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ros-jazzy-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Gazebo Harmonic
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-sim \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# TF
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ros-jazzy-tf2-tools \
    ros-jazzy-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Navigation utilities
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ros-jazzy-xacro \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-robot-state-publisher \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Useful CLI tools
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    iputils-ping \
    net-tools \
    htop \
    unzip \
    tree \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Workspace
# -----------------------------------------------------------------------------
WORKDIR /root/ros2_ws

RUN mkdir -p src

# -----------------------------------------------------------------------------
# Source ROS automatically
# -----------------------------------------------------------------------------
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

CMD ["bash"]