## Train

```bash
python train.py
```

## Docker build

```bash
docker build . -t deeplearning
```

## Start Docker container

```bash
docker run  -itd --name oasis_trainer \
            -v <local-dataset-path>:/mnt/oasis \
            -e "KAGGLE_USERNAME=*************" \
            -e "KAGGLE_KEY=******************" \
            -m 32g --cpus=2\
            deeplearning
```

### Example

```bash
docker run -itd --name oasis_trainer -v "D:\CGC work\Alziehmer Disease\PhdNotebook\:/mnt/oasis" -e "KAGGLE_USERNAME=guxxxx1" -e "KAGGLE_KEY=0513c7fabxxxxxxx5d09c003cd961532" -m 14g --cpus=2  deeplearning
```

## Start training inside Docker container

- open docker-desktop application
- locate the running container `oasis_trainer`
- Click 'Exec'
- Run `python train.py`

## Improvement tips for docker building

> **Note:** Run the following code inside docker only

```bash
pip download -d /mnt/oasis/py_packages h5py==3.11.0
pip download -d /mnt/oasis/py_packages keras==2.13.1  tensorflow==2.13.0 tensorflow-datasets==4.9.2 
pip download -d /mnt/oasis/py_packages torch==2.4.1 torchvision==0.19.1 
pip download -d /mnt/oasis/py_packages opendatasets==0.1.22 seaborn==0.13.2 pillow==10.2.0 scikit-learn==1.3.2 pandas==2.0.3 
pip download -d /mnt/oasis/py_packages opencv-python==4.10.0.84 
```