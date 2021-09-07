FROM nvidia/cuda:11.2.2-cudnn8-devel
MAINTAINER Christian Eder (christian.eder@tuwien.ac.at)

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Linux package install
RUN sed 's/main$/main universe/' -i /etc/apt/sources.list
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated --fix-missing \
      build-essential \
      autoconf \
      automake \
      git \
      wget \
      usbutils \
      unzip \
      vim \
      software-properties-common \
      libxext-dev \
      libxrender-dev \
      libxslt1.1 \
      libxtst-dev \
      libgtk2.0-0 \
      libcanberra-gtk-module \
      tmux \
      && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN apt-get update
SHELL ["/bin/bash", "-c"]

RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev tmux git

# Fixes shared memory error in docker
RUN echo "export QT_X11_NO_MITSHM=1" >> ~/.bashrc

# Python 3
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python3-dev \
     python3-numpy \
     python3-pip \
     && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools

RUN pip3 install torch==1.4.0 torchvision==0.5.0 torchaudio

#RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
#RUN pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl 
RUN pip3 install plyfile pillow
RUN pip3 install scipy==1.6.1
RUN apt-get update
RUN apt-get install libsparsehash-dev

# ROS
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
      python3-rosinstall \
      python3-rosinstall-generator \
      python3-wstool \
      python3-rosdep \
      python3-vcstools \
      && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
    rosdep update

# catkin tools
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python3-catkin-tools \
     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
      ros-noetic-ros-core \
      ros-noetic-libuvc-camera \
      ros-noetic-image-view \
      ros-noetic-cv-bridge \
      ros-noetic-cv-camera \
      ros-noetic-actionlib \
      ros-noetic-ros-numpy \
      && rm -rf /var/lib/apt/lists/*


RUN sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# install python packages
RUN pip3 install --upgrade osrf-pycommon
RUN pip3 install --upgrade rospkg catkin_pkg  #opencv-contrib-python3 empy

# for ros environments
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Prepare catkin build
RUN mkdir -p ~/catkin_build_ws/src

# Prepare share folder
RUN mkdir -p ~/share

# Run catkin build
RUN /bin/bash -c  '. /opt/ros/noetic/setup.bash; cd ~/catkin_build_ws; catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so; catkin build'

# source the catkin workspace
RUN echo "source ~/catkin_build_ws/devel/setup.bash" >> ~/.bashrc

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN git clone --branch robot_experiments https://username:password@rgit.acin.tuwien.ac.at/edith-langer/sparseconvnet.git ~/catkin_build_ws/src/sparseconvnet_ros
RUN cd ~/catkin_build_ws/ && catkin build
RUN source ~/catkin_build_ws/devel/setup.bash
#RUN cd ~/catkin_build_ws/src/sparseconvnet_ros && bash develop.sh
#RUN cd ~/catkin_build_ws/src/sparseconvnet_ros && rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
#RUN cd ~/catkin_build_ws/src/sparseconvnet_ros && python3 setup.py develop
#RUN cd ~/catkin_build_ws/src/sparseconvnet_ros && python3 examples/hello-world.py

WORKDIR /root

# setup entrypoint
COPY ./ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
