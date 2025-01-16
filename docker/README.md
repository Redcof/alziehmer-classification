## On host machine
```
cd docker
docker build . -t deeplearning
docker run  -itd --name oasis_trainer \
            -v <local-dataset-path>:/mnt/oasis \
            -e "KAGGLE_USERNAME=<kaggel-username>" \
            -e "KAGGLE_KEY=<kaggel-key>" \
            -m 32g --storage-opt size=100G --cpus=2\
            deeplearning

# example
            
# docker run -itd --name oasis_trainer -v C:\Users\phduser\Documents\experiment:/mnt/oasis -e "KAGGLE_USERNAME=gagan" -e "KAGGLE_KEY=123abc1256780956acedbacbed234523" deeplearning
```

## Inside docker
```
python train.py
```