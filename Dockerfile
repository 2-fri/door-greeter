# Setup
FROM ros:humble
ENV DEBIAN_FRONTEND=noninteractive

# Get Packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    portaudio19-dev \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install ultralytics
RUN pip install facenet_pytorch
RUN pip install sqlite_vec
RUN pip install groq
RUN pip install dotenv
RUN apt-get install portaudio19-dev
RUN pip install SpeechRecognition[audio]
RUN pip install "numpy<2"

# Add Package
WORKDIR /door-greeter/src
COPY . /door-greeter/src

WORKDIR /door-greeter
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install