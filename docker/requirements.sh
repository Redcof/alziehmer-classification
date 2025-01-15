set -x
set -e
pip install --upgrade pip
pip install h5py
# pip install keras==2.13.1  tensorflow==2.13.0 tensorflow-datasets==4.9.2 --trusted-host files.pythonhosted.org
pip install opendatasets==0.1.22
pip install seaborn==0.13.2 pillow==10.2.0 scikit-learn==1.3.2
pip install opencv-python==4.10.0.84
pip install torch torchvision
# python -m tensorflow tensorflow_datasets torch torchvision cv2 pil seaborn
set +x