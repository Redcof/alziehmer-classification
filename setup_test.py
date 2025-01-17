import keras, tensorflow, tensorflow_datasets, torch, torchvision, cv2, PIL, seaborn, pandas
# pre-download the model
keras.applications.MobileNet(weights='imagenet')