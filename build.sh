source /opt/ros/humble/setup.bash
colcon build
source ~/door-greeter/install/setup.bash
export LD_LIBRARY_PATH=$HOME/door-greeter/external/install/lib:$LD_LIBRARY_PATH
export GROQ_API_KEY="$(cat .groq_key)"