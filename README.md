### Number Plate Recognition

#### A yolov4 tiny model trained using Darknet and converted to pytorch to improve efficency
 
### Environment

Developed in an Arch environment but should work on Windows and Mac as well

<img src="/screens/env.jpg" style="float: center; margin-right: 10px;" width="1000"/>

### Installation

#### Pytorch

#### Install Dependencies

Common
```bash
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
```

On Linux
```bash
# Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda102  # or [ magma-cuda101 | magma-cuda100 | magma-cuda92 ] depending on your cuda version
```

On MacOS
```bash
# Add these packages if torch.distributed is needed
conda install pkg-config libuv
```

On Windows
```bash
# Add these packages if torch.distributed is needed.
# Distributed package support on Windows is a prototype feature and is subject to changes.
conda install -c conda-forge libuv=1.39
```

#### Get the PyTorch Source
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

#### Install PyTorch
On Linux
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```
On macOS
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
```
[**More installation instructions**](https://github.com/pytorch/pytorch#from-source)

#### OpenCV
On Linux

install deoendencies

```bash
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
```

optional

```bash
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-devlibtiff-dev libjasper-dev libdc1394-22-dev
```

Build

```bash
mkdir ~/src
cd ~/src
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=ON ..
make -j$(nproc)
sudo make install
```

#### Openalpr

```bash

Install openALPR for Ubuntu (Much easier and had a higher success rate)

```bash
# Install prerequisites
sudo apt-get install libopencv-dev libtesseract-dev git cmake build-essential libleptonica-dev
sudo apt-get install liblog4cplus-dev libcurl3-dev

# If using the daemon, install beanstalkd
sudo apt-get install beanstalkd

# Clone the latest code from GitHub
git clone https://github.com/openalpr/openalpr.git

# Setup the build directory
cd openalpr/src
mkdir build
cd build

# setup the compile environment
RUN cmake -D WITH_BINDING_JAVA=OFF \
  -D WITH_DAEMON=OFF \
  -D WITH_GPU_DETECTOR=ON \
  -D CMAKE_INSTALL_PREFIX:PATH=/usr \
  -D CMAKE_INSTALL_SYSCONFDIR:PATH=/etc ..

# compile the library
make -j4

# Install the binaries/libraries to your local system (prefix is /usr)
sudo make install

# Test the library
wget http://plates.openalpr.com/h786poj.jpg -O lp.jpg
alpr lp.jpg
```

Once you've completed the steps, go to the git repo you've cloned to your machine during the above installation and install the python bindings. runtime_data folder is also available in this repo, which will be helpful for next step.

```bash
cd openalpr/src/bindings/python/
sudo python3 setup.py install
```
### Opimizations

After this step follow these to optimize openalpr's ocr for some perfomance gains

You should modify file /etc/openalpr/openalpr.conf example below
```python
; This configuration file overrides the default values specified 
; in /usr/share/openalpr/config/openalpr.defaults.conf
hardware_acceleration = 1
gpu_id = 0
gpu_batch_size = 10
```
You should modify file /usr/share/openalpr/openalpr.defaults.conf example below
```python
max_detection_input_width = 800
max_detection_input_height = 600

detector = lbpgpu

skip_detection = 1

max_plate_angle_degrees = 30
```
change the resolution in cfg/yolov4.cfg

```bash

[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch=64
subdivisions=16
width=608      #change the height and width lower resolutions
height=608     #improve perfomance and higher resolutions 
channels=3     #improve accuracy
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

```
## Credits

[**Darknet**](https://github.com/AlexeyAB/darknet)
[**pytorch-YOLOv4**](https://github.com/Tianxiaomo/pytorch-YOLOv4)
[**Pytorch**](https://pytorch.org/)
[**OpenCV**](https://opencv.org/)
[**Openalpr**](https://github.com/openalpr/openalpr)
