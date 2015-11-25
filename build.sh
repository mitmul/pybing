#! /bin/bash

# Build
if [ ! -d "build" ]; then
  mkdir build
fi

cd build

cmake ../
make
touch __init__.py

git clone https://github.com/Itseez/opencv_contrib
mv opencv_contrib/modules/saliency/samples/ObjectnessTrainedModel ./
rm -rf opencv_contrib

cd ..

wget http://farm1.static.flickr.com/121/278839518_140821637d.jpg -O sample.jpg
python scripts/test_bing.py
