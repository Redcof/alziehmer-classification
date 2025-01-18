#!/bin/bash

set -e

if [ $"(ls -A './centos_amd64_deps')" ];
then
    echo "Empty whl folder";
    export NOINDEX='';
else
    echo "Non-Empty whl folder";
    export NOINDEX='--no-index --find-links=centos_amd64_deps';
fi;

pip install --upgrade pip
pip install h5py==3.11.0 $NOINDEX
pip install keras==2.13.1  tensorflow==2.13.0 tensorflow-datasets==4.9.2 $NOINDEX
pip install torch==2.4.1 torchvision==0.19.1 $NOINDEX
pip install opendatasets==0.1.22 seaborn==0.13.2 pillow==10.2.0 scikit-learn==1.3.2 pandas==2.0.3 $NOINDEX
pip install opencv-python==4.10.0.84 $NOINDEX
