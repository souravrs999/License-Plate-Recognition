### Number Plate Recognition

#### A yolov4 tiny model trained using Darknet and converted to pytorch to improve efficency
 
### Screenshots
<img src="/screens/sc1.png" style="float: left; margin-right: 10px;" width="500"/><img src="/screens/sc2.png" style="float: left; margin-right: 10px;" width="500"/>

### Installation

This code was written and tested on a Linux machine but the installation steps will be similar for Windows an MacOS

Clone the darknet repository

```bash
git clone https://github.com/AlexeyAB/darknet.git
```
Download these files and copy them to the cloned darknet repository

Open the Makefile and change these lines as folows

```python
GPU=1 # 0 to 1 if you have a GPU
CUDNN=1 # 0 to 1 if you want to use CUDNN
CUDNN_HALF=0 # 0 to 1 if you have a higher end GPU
OPENCV=1 # 0 to 1 to use opencv will not work if you dont set it to 1
AVX=1 # 0 to 1 useful if you need to run it in a CPU
OPENMP=0 # 0 to 1 (user preference)
LIBSO=1 # 0 to 1 Important the code wont run without it will generate a libdarknet.so
ZED_CAMERA=0 # 0 to 1 (user preference)
ZED_CAMERA_v2_8=0 # 0 to 1 (user preference)

# if using a Jetson Nano board uncomment the following(for jetson TX1)
# ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

```
Run make

```bash
make -j4
```
Darknet will be compiled with GPU support

### OpenALPR https://github.com/openalpr/openalpr

This code uses openALPR OCR to read the characters from the detected number plates

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

## Credits

AlexyAB for darknet

