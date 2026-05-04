pip install ultralytics
pip install facenet_pytorch
pip install sqlite_vec
pip install groq
pip install SpeechRecognition[audio]
pip install pyttsx3

cd ~/door-greeter
mkdir -p external
cd external

curl -LO https://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz
tar -xvzf pa_stable_v190700_20210406.tgz
cd portaudio
./configure --prefix=$HOME/door-greeter/external/install
make
make install

export PORTAUDIO_PREFIX=$HOME/door-greeter/external/install
export CFLAGS="-I$PORTAUDIO_PREFIX/include"
export LDFLAGS="-L$PORTAUDIO_PREFIX/lib"
export LD_LIBRARY_PATH="$PORTAUDIO_PREFIX/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$PORTAUDIO_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

source /opt/ros/humble/setup.bash

cd ~/door-greeter/src
git clone https://github.com/ckennedy2050/Azure_Kinect_ROS2_Driver.git

cd ~/door-greeter

pip install --no-binary :all: pyaudio

pip install "numpy<2"