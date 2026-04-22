docker build -t my_ros2_image .
docker run -it \
  --device /dev/snd \
  --group-add audio \
  my_ros2_image
source install/setup.bash