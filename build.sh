#! /bin/bash

######### PLEASE MODIFY #########

TBB_INSTALL_DIR=/opt/intel/tbb
TBB_ARCH=intel64
TBB_COMPILER=gcc4.4

OPENCV_SHARE_DIR=$HOME/Libraries/opencv/install/share/OpenCV
OPENCV_CONTRIB_DIR=$HOME/Libraries/opencv_contrib

BOOST_INSTALL_DIR=$HOME/Libraries/boost_1_58_0/install
BOOST_NUMPY_INSTALL_DIR=$HOME/Libraries/Boost.NumPy/install

#################################

# Build
if [ ! -d "build" ]; then
  mkdir build
fi

cd build

export TBB_INSTALL_DIR=$TBB_INSTALL_DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TBB_INSTALL_DIR/lib/$TBB_ARCH/$TBB_COMPILER

cmake \
-DOpenCV_DIR=$OPENCV_SHARE_DIR \
-DBOOST_ROOT=$BOOST_INSTALL_DIR \
-DBoost_NumPy_INCLUDE_DIR=$BOOST_NUMPY_INSTALL_DIR/include \
-DBoost_NumPy_LIBRARY_DIR=$BOOST_NUMPY_INSTALL_DIR/lib \
../

make

cd ..

cp -r $OPENCV_CONTRIB_DIR/modules/saliency/samples/ObjectnessTrainedModel build/
wget http://farm1.static.flickr.com/121/278839518_140821637d.jpg -O sample.jpg
