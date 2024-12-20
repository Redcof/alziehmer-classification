import os
import pathlib
import random
import warnings

import cv2
import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset


class OASISTorchDataset(Dataset):
    BINARY_TASK = "binary"
    CATEGORY_TASK = "categorical"
    VOLUME_DEPTH = 61

    def __init__(self, path, task_type, image_size, transforms, nch=1, seed=37, class_names=None, dtype="float32"):
        self.path = pathlib.Path(path)
        self.classnames = [i for i in os.listdir(path)]
        assert len(image_size) == 2, "Image size must be a tuple of two integers"
        self.image_size = image_size
        self.dtype = dtype
        self.transforms = transforms
        if class_names is not None:
            self.classnames = [c for c in self.classnames if c in class_names]
        self.num_classes = len(self.classnames)
        if task_type == self.BINARY_TASK:
            self.num_classes = 1
        self.task_type = task_type
        assert nch in (1, 3), "Supported channels are 1(greyscale) and 3(rgb)"
        self.nch = nch
        if self.task_type == self.BINARY_TASK and len(self.classnames) >= 2 and class_names is None:
            raise Exception(f"When `class_names` is not specified, "
                            f"only 2 classes are allowed in the data-directory, but {len(self.classnames)} found.")
        self.items = []
        self._load(seed)

    @staticmethod
    def _group_images_by_subject(image_dir):
        """Groups images by subject ID based on their filenames.
        Args:
            image_dir: Path to the directory containing the images.
        Returns:
            A dictionary where keys are subject IDs and values are lists of image filenames.
        """
        image_files = os.listdir(image_dir)
        subject_groups = {}
        for filename in image_files:
            # Extract subject ID from the filename (adjust the pattern as needed)
            subject_id = "_".join(filename.split('_')[0:-1])
            if subject_id not in subject_groups:
                subject_groups[subject_id] = []
            subject_groups[subject_id].append(os.path.join(image_dir, filename))
        print("For", len(subject_groups.keys()), "patients,", len(image_files), f"images scanned from '{image_dir}'")
        return subject_groups

    def item_generator(self):
        """
        :return: yield MRI slices as list and encoded label
        """
        for class_dir in self.classnames:
            grouped_files = self._group_images_by_subject((self.path / class_dir).resolve())
            for slices, label in zip(grouped_files.values(), [class_dir] * len(grouped_files.values())):
                lbl = self._encode_labels(label)
                yield slices, lbl

    def volume_generator(self):
        """
        :return: yield MRI slices as 3D RGB image volume and encoded label
        """
        for class_dir in self.classnames:
            grouped_files = self._group_images_by_subject((self.path / class_dir).resolve())
            for slices, label in zip(grouped_files.values(), [class_dir] * len(grouped_files.values())):
                vol = self._load_volume(slices)
                lbl = self._encode_labels(label)
                yield vol, lbl

    def _encode_labels(self, labels):
        """
        :param labels: this could be a string or list of strings
        :return: encoded binary tensor of [1,0,1,...] or [[0,1],[1,0],[0,1],...]
                for categorical
        """
        is_one_element = False
        if isinstance(labels, str):
            labels = [labels]
            is_one_element = True
        enc_label = []
        for lbl in labels:
            if lbl not in self.classnames:
                warnings.warn(f"'{lbl}' label is unknown")
            if self.task_type == self.BINARY_TASK:
                enc_label.append(self.classnames.index(lbl))
            else:
                enc_label.append([lbl == cls_nm for cls_nm in self.classnames])
        if is_one_element:
            return np.array(enc_label[0]).astype(self.dtype)
        else:
            return np.array(enc_label).astype(self.dtype)

    def _decode_labels(self, probabilities, probability_threshold=0.5):
        """
        :param probabilities: batch of the network output(probabilities)
        :param probability_threshold: to decide the class
        :return: decode [probas1, probas2, ...] to tensor of [cls1, cls2,...]
        """
        assert len(probabilities.shape) == 2, f"`probabilities` shape must be two dimensional"
        if self.task_type == self.BINARY_TASK:
            assert probabilities.shape[1] == 1, f"`probabilities` shape must be (batch_size, 1)"
            binaries = probabilities >= probability_threshold
            return list(map(lambda oi: self.classnames[int(oi)], binaries))
        else:
            assert probabilities.shape[1] == len(self.classnames), (f"`probabilities` shape must be"
                                                                    f" (batch_size, num_classes)")
            probability_threshold = [probability_threshold] * len(self.classnames)
            indices = (probabilities >= probability_threshold).numpy().argmax(axis=1)
            string_array = np.array(self.classnames)
            return string_array[indices].tolist()

    def _load_volume(self, image_files: list, label=None):
        """Loads a 3D volume from a directory of JPEG images.
        Args:
            image_files: list of files
        Returns:
            A 3D representing the 3D volume
        """
        image_volume = []
        for image_path in image_files:
            image_data = cv2.imread(image_path)
            if self.nch == 1:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            else:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            # resize
            image_data = cv2.resize(image_data, self.image_size)
            # transforming images
            image_data = self.transforms(image_data.astype("uint8"))
            # creating volume
            image_volume.append(image_data)
        volume = np.stack(image_volume, axis=0).astype(self.dtype)
        if label is None:
            return volume
        else:
            return volume, label

    def _load(self, seed):
        random.seed(seed)
        self.items = list(self.item_generator())
        random.shuffle(self.items)
        return self

    def __len__(self, ):
        return len(self.items)

    def __getitem__(self, idx):
        return self._load_volume(*self.items[idx])


def torch_to_tf(torch_dataset, task_type, image_size, nch=3, dtype="float32"):
    def torch_data_gen():
        for img, lbl in torch_dataset:
            yield tf.convert_to_tensor(img), tf.convert_to_tensor(lbl)

    input_shape = (61, *image_size, nch)
    label_shape = (1,) if task_type == OASISTorchDataset.BINARY_TASK else (4,)
    input_datatype = (getattr(tf, dtype), getattr(tf, dtype))
    dataset = tf.data.Dataset.from_generator(torch_data_gen,
                                             output_types=input_datatype,
                                             output_shapes=(input_shape, label_shape)
                                             )
    return dataset


class OASIS_TF:
    def __init__(self):
        self.classnames = None
        self.num_classes = None
        self.num_items = None

    def tf_oasis_load_dataset(
            self,
            directory,
            transforms,
            label_mode="category",
            class_names=None,
            color_mode="rgb",
            batch_size=32,
            image_size=(256, 256),
            seed=None,
            validation_split=0.2,
    ):
        assert color_mode in ("greyscale", "rgb"), color_mode
        assert label_mode in (OASISTorchDataset.BINARY_TASK, OASISTorchDataset.CATEGORY_TASK), label_mode
        nch = 3 if color_mode == "rgb" else 1
        dtype = "float32"
        # load torch dataset
        torch_dataset = OASISTorchDataset(directory,
                                          task_type=label_mode,
                                          image_size=image_size,
                                          transforms=transforms,
                                          nch=nch,
                                          seed=seed,
                                          class_names=class_names,
                                          dtype=dtype)
        self.classnames = torch_dataset.classnames
        self.num_classes = torch_dataset.num_classes
        self.num_items = len(torch_dataset)
        print("Number of patients:", self.num_items)
        print("Class Names:", self.classnames)
        print("Number of classes:", self.num_classes)
        print(f"Minimum data waste for {batch_size=} is", self.num_items % batch_size)
        # convert torch to tf dataset
        dataset = torch_to_tf(torch_dataset,
                              task_type=label_mode,
                              image_size=image_size,
                              nch=nch, dtype="float32")
        # calculate split sizes
        dataset_size = len(torch_dataset)  # Assuming dataset has a defined length
        train_size = int(1. - validation_split * dataset_size)
        # split
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        # batch
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        return train_dataset, val_dataset

    def __len__(self):
        return self.num_items
