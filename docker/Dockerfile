# ROS 2 Humble + MoveIt 2 + this workspace.
#
#   docker build -f docker/Dockerfile -t franka_pick_place .
#   xhost +local:docker
#   docker run --rm -it --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix franka_pick_place \
#     ros2 launch grasp_planning pick_place.launch.py
#
# Build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 for a GPU image.
FROM ros:humble-ros-base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-colcon-common-extensions python3-pytest \
      ros-humble-moveit ros-humble-moveit-resources-panda-moveit-config \
      ros-humble-realsense2-camera ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX} \
    && pip3 install --no-cache-dir "numpy<2" pandas tabulate

WORKDIR /ws
COPY src/ src/
RUN apt-get update && rosdep update && rosdep install --from-paths src --ignore-src -y \
    && rm -rf /var/lib/apt/lists/*
RUN . /opt/ros/humble/setup.sh && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN echo "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash" >> /root/.bashrc
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]
