# ==========================================================
import os

import cv2
import numpy as np
import torchvision.transforms

from oasis_torch_dataset import OASISTFDataset, OASISTorchDataset

# from rich import print

# DO NOT TOUCH BELOW ==================================
FULL_DATASET = 0
FULL_BINARY_DATASET = 1
MY_DATASET = 2
MY_BIN_DATASET = 3

# DO NOT TOUCH - END ===================================

# select dataset
SELECTED_DATASET = MY_DATASET  # CHANGE HERE

# DO NOT TOUCH BELOW ==================================
FULL_DATASET_PATH = r'D:\CGC work\Alziehmer Disease\PhdNotebook\OASIS dataset\OASIS dataset\Data'
FULL_BINARY_DATASET_PATH = r'D:\CGC work\Alziehmer Disease\PhdNotebook\OASIS dataset Binary\OASIS dataset\Data'
MY_DATASET_PATH = r'/Users/soumensardar/Downloads/OASIS/'
MY_DATASET_BINARY_PATH = r'/Users/soumensardar/Downloads/OASIS-binary/'

# while adding more datasets, make sure to add tflog directory
datasetdir, dataset_name = [(FULL_DATASET_PATH, 'tflogs_3d'),
                            (FULL_BINARY_DATASET_PATH, 'tflogs_binary_3d'),
                            (MY_DATASET_PATH, 'tflogs_my_3d'),
                            (MY_DATASET_BINARY_PATH, 'tflogs_mybin_3d'),
                            ][SELECTED_DATASET]
# DO NOT TOUCH - END ===================================
datasetdir, dataset_name
assert os.path.exists(datasetdir), f"Dataset path is incorrect {datasetdir}"

random_seed = 37
resume_training_timestamp = None  # CHANGE HERE
max_epoch = 50  # CHANGE HERE
batch_size = 13  # CHANGE HERE
image_size = 128
color_mode = 'rgb'  # grayscale
n_ch = dict(rgb=3, grayscale=1)[color_mode]
monitor = "f1_score"
initial_threshold = 0.5
mode = "max"
freq = "epoch"
initial_epoch = 0
learning_rate = 0.001
enable_class_weight = False
ablation_study_size = 0  # CHANGE HERE
task_type = "categorical"  # "categorical"
# Link: https://www.tensorflow.org/api_docs/python/tf/keras/losses#functions
if task_type == "binary":
    activation = "sigmoid"
    loss_function = "binary_crossentropy"
else:
    activation = "softmax"
    loss_function = "categorical_crossentropy"
lr_schedular = False
resume_checkpoint_path = None
# Link: https://keras.io/api/applications/
model_name = "MobileNet"  # CHANGE HERE !!!!!!!
# add a comment about what changes you have done just now before running the training
what_changed = f"Training with {model_name=} {task_type=} {lr_schedular=} {activation=} {loss_function=}"
# ==========================================================

if __name__ == '__main__':
    def bilateral_filter(pil_image):
        return cv2.bilateralFilter(np.array(pil_image), 15, 75, 75)


    def img_reshape(tensor):
        return tensor.view(image_size, image_size, n_ch)


    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Lambda(bilateral_filter),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(img_reshape),
        torchvision.transforms.Normalize(mean=0., std=1.),
    ])
    OASISTorchDataset.VOLUME_DEPTH = 16
    odd = OASISTFDataset()
    train_ds, val_ds, test_ds = odd.tf_oasis_load_dataset(
        datasetdir,
        transforms=transforms,
        label_mode=task_type,
        class_names=None,
        color_mode=color_mode,
        batch_size=batch_size,
        ablation=ablation_study_size,
        image_size=(image_size, image_size),
        seed=random_seed,
        split_ratio_100=(70, 20, 10),
    )
    odd.sample_dataset.get_volume_for_image(
        r'/Users/soumensardar/Downloads/OASIS/Mild Dementia/OAS1_0028_MR1_mpr-1_127.jpg')
    for im, lb in train_ds:
        print(im.shape, odd.decode_labels(lb))
