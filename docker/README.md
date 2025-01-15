```
docker build . -t python38
docker run -it --name oasis_trainer -v ~/Downloads:/mnt/oasis python38
docker run -it --name oasis_trainer -v ~/Downloads:/mnt/oasis deeplearning
python train.py
```