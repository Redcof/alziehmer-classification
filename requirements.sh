#!/bin/bash

set -e
set -x

python -m pip install --upgrade pip

# Define a function with parameters
resolve_dependencies() {
    local command="$1"
    local index="$2"
    pip $command h5py==3.11.0 $index
    pip $command keras==2.13.1  tensorflow==2.13.0 tensorflow-datasets==4.9.2 $index
    pip $command torch==2.4.1 torchvision==0.19.1 $index
    pip $command opendatasets==0.1.22 seaborn==0.13.2 pillow==10.2.0 scikit-learn==1.3.2 pandas==2.0.3 $index
    pip $command opencv-python==4.10.0.84 $index
}

# Directory to check
DIR="centos_amd64_deps"

# Check if directory exists
if [ -d "$DIR" ]; then
    # Check if directory is empty
    if [ "$(ls -A "$DIR")" ]; then
        echo "The directory '$DIR' is not empty, installing libraries form the local cache..."
    else
        echo "The directory '$DIR' is empty, installing libraries form the internet..."
        resolve_dependencies "download -d $DIR" ""
    fi
else
    echo "The directory '$DIR' does not exist, installing libraries form the internet..."
    mkdir $DIR
    resolve_dependencies "download -d $DIR" ""
fi
resolve_dependencies "install" "--no-index --find-links=$DIR"
