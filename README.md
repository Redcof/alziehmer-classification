## Train

```
python train.py
```

## Docker build

```
docker build . -t deeplearning
```

## Start Docker container

```
docker run  -itd --name oasis_trainer \
            -v <local-dataset-path>:/mnt/oasis \
            -e "KAGGLE_USERNAME=<kaggel-username>" \
            -e "KAGGLE_KEY=<kaggel-key>" \
            -m 32g --cpus=2\
            deeplearning
```

### Example

```
# docker run -itd --name oasis_trainer -v C:\Users\phduser\Documents\experiment:/mnt/oasis -e "KAGGLE_USERNAME=gagan" -e "KAGGLE_KEY=123abc1256780956acedbacbed234523" deeplearning
```

## Start training inside Docker container

- open docker-desktop application
- locate the running container `oasis_trainer`
- Click 'Exec'
- Run `python train.py`
