import os
import pathlib
import random
import warnings

import cv2
import numpy as np
import tensorflow as tf
import torchvision.transforms
from torch.utils.data import Dataset


class OASISTorchDataset(Dataset):
    BINARY_TASK = "binary"
    CATEGORY_TASK = "categorical"
    VOLUME_DEPTH = 61  # KNOWN DURING EDA: per patient we have 61 MRI slices

    def get_volume_for_image(self, oasis_image_path):
        """
        For a given image this function automatically determine the slices and return the volume and actual label
        :param oasis_image_path:
        :return:
        """

        oasis_image_path = pathlib.Path(oasis_image_path)
        class_name = oasis_image_path.parent.name
        # OAS1_0001_MR1_mpr-1_101.jpg
        fp, sp = oasis_image_path.name.split("-")  # [OAS1_0001_MR1_mpr, 1_101.jpg]
        spfp, spsp = sp.split("_")  # [1, 101.jpg]
        pattern = fp + "-" + spfp + "*"
        images = list(oasis_image_path.parent.glob(pattern))
        images = self._sample_from(images, OASISTorchDataset.VOLUME_DEPTH)
        lbl = self.encode_labels(class_name)
        return self.load_volume(images), class_name, lbl

    def __init__(self, path, task_type, image_size, transforms, split_name, batch_size, ablation=0, nch=3, seed=37,
                 splits=(70, 20, 10),
                 class_names=None,
                 dtype="float32", verbose=0):
        assert len(image_size) == 2, "Image size must be a tuple of two integers"
        assert split_name in ("train", "validation", "test")
        assert nch in (1, 3), "Supported channels are 1(greyscale) and 3(rgb)"
        assert len(splits) == 3 and sum(
            splits) == 100, f"train-validation-test splits must accumulate to 100, but {splits=} given."
        self.path = pathlib.Path(path)
        self.verbose = verbose
        # get class names
        self.classnames = self._load_class_names()
        if class_names is not None:
            self.classnames = [c for c in self.classnames if c in class_names]
        if task_type == self.BINARY_TASK and len(self.classnames) >= 2 and class_names is None:
            raise Exception(f"When `class_names` is not specified, "
                            f"only 2 classes are allowed in the data-directory, but {len(self.classnames)} found.")
        self.num_classes = len(self.classnames)
        if task_type == self.BINARY_TASK:
            self.num_classes = 1

        self.image_size = image_size
        self.dtype = dtype
        self.batch_size = batch_size
        self.transforms = transforms if transforms else torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0., std=1.),
        ])
        self.split_name = split_name
        self.task_type = task_type
        self.nch = nch
        self.splits = splits
        self.items = []
        self.ablation = ablation
        # load dataset
        self._load(seed)

    def _load_class_names(self):
        return [i for i in os.listdir(self.path) if i != ".DS_Store"]

    def _group_images_by_subject(self, image_dir):
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
            # OAS1_0001_MR1_mpr-1_101.jpg
            subject_id = "_".join(filename.split('_')[0:-1])
            if subject_id not in subject_groups:
                subject_groups[subject_id] = []
            subject_groups[subject_id].append(os.path.join(image_dir, filename))
        if self.verbose:
            print("For", len(subject_groups.keys()), "patients,", len(image_files),
                  f"images scanned from '{image_dir}'")
        return subject_groups

    def item_generator(self):
        """
        :return: yield MRI slices as list and encoded label
        """
        for class_dir in self.classnames:
            grouped_files = self._group_images_by_subject((self.path / class_dir).resolve())
            for slices, label in zip(grouped_files.values(), [class_dir] * len(grouped_files.values())):
                lbl = self.encode_labels(label)
                yield self._sample_from(slices, OASISTorchDataset.VOLUME_DEPTH), lbl

    @staticmethod
    def _sample_from(lst, n):
        if n > len(lst):
            raise ValueError("Sample size cannot be greater than the list size.")

            # Calculate step size for approximately even distribution
        step_size = len(lst) // n

        # Generate indices with approximately even spacing
        indices = [i * step_size for i in range(n)]

        # Adjust last index to ensure it's within bounds
        indices[-1] = min(indices[-1], len(lst) - 1)
        return [lst[i] for i in indices]

    def encode_labels(self, labels):
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

    def decode_labels(self, probabilities: np.ndarray, probability_threshold=0.5):
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
            indices = (probabilities >= probability_threshold).argmax(axis=1)
            string_array = np.array(self.classnames)
            return string_array[indices].tolist()

    def load_volume(self, image_files: list, label=None):
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
            if self.transforms is not None:
                image_data = self.transforms(image_data.astype("uint8"))
            # creating volume
            image_volume.append(image_data)
        volume = np.stack(image_volume, axis=0).astype(self.dtype).reshape((OASISTorchDataset.VOLUME_DEPTH,
                                                                            *self.image_size, self.nch))
        if label is None:
            return volume
        else:
            return volume, label

    def _collect_items(self):
        return list(self.item_generator())

    def _get_split_sizes(self, total_items):
        size = total_items
        # convert [0,100] scale to [0,1]
        train_ratio = self.splits[0] / 100
        val_ratio = self.splits[1] / 100
        test_ratio = self.splits[2] / 100
        # calculate no. of batches possible
        no_batches = (size // self.batch_size)
        # calculate no. of batches per split
        train_batches = int(round(no_batches * train_ratio))
        val_batches = int(round(no_batches * val_ratio))
        test_batches = int(round(no_batches * test_ratio))
        # calculate residual batches
        residual_batches = train_batches + val_batches + test_batches - no_batches
        # adjust residual batches to/from train_batches
        if residual_batches < 0:
            # if not all batches are used, add remaining batches to train set
            test_batches += -residual_batches
        if residual_batches > 0:
            # if more batches are used, remove extra batches from train set
            test_batches -= residual_batches
        # now calculate item count per split form batch count
        train_size = train_batches * self.batch_size
        val_size = val_batches * self.batch_size
        test_size = test_batches * self.batch_size
        if self.verbose:
            print(f"Dataset {train_size=}, {val_size=}, and {test_size=}")
        return train_size, val_size, test_size

    def _split(self, items):
        train_size, val_size, test_size = self._get_split_sizes(len(items))
        if self.split_name == "train":
            # train
            self.items = items[:train_size]
        elif self.split_name == "validation":
            # validation
            self.items = items[train_size: train_size + val_size]
        else:
            # test
            self.items = items[train_size + val_size:]
        if self.ablation == 0:
            if len(self.items) % self.batch_size != 0:
                warnings.warn(f"For {self.split_name=} {len(self.items) % self.batch_size} patient(s) MRI(s) "
                              f"({(len(self.items) % self.batch_size) * OASISTorchDataset.VOLUME_DEPTH}) slices are"
                              f" not in use")

        return self.items

    def _load(self, seed):
        items = self._collect_items()
        # shuffle
        random.seed(seed)
        random.shuffle(items)
        # ablation
        if self.ablation:
            items = items[:self.ablation]
        # split
        self.items = self._split(items)
        return self

    def __len__(self, ):
        return len(self.items)

    def __getitem__(self, idx):
        return self.load_volume(*self.items[idx])


class OASISTFDataset:
    def __init__(self):
        self.sample_dataset: OASISTorchDataset = None
        self.classnames = None
        self.num_classes = None
        self.num_items = None
        self.recommended_batches = None

    @staticmethod
    def torch_to_tf(torch_dataset):
        def torch_data_gen():
            for img, lbl in torch_dataset:
                yield tf.convert_to_tensor(img), tf.convert_to_tensor(lbl)

        input_shape = (torch_dataset.VOLUME_DEPTH, *torch_dataset.image_size, torch_dataset.nch)
        label_shape = (torch_dataset.num_classes,)
        input_datatype = (getattr(tf, torch_dataset.dtype), getattr(tf, torch_dataset.dtype))
        dataset = tf.data.Dataset.from_generator(torch_data_gen,
                                                 output_types=input_datatype,
                                                 output_shapes=(input_shape, label_shape)
                                                 )
        return dataset

    def tf_oasis_load_dataset(
            self,
            directory,
            transforms,
            label_mode="category",
            class_names=None,
            color_mode="rgb",
            batch_size=32,
            ablation=0,
            image_size=(256, 256),
            seed=None,
            split_ratio_100=(70, 20, 10),
    ):
        assert color_mode in ("greyscale", "rgb"), color_mode
        assert label_mode in (OASISTorchDataset.BINARY_TASK, OASISTorchDataset.CATEGORY_TASK), label_mode
        assert len(split_ratio_100) == 3 and sum(
            split_ratio_100) == 100, f"train-validation-test splits must accumulate to 100, but {split_ratio_100} given"
        # determine number of channels
        nch = 3 if color_mode == "rgb" else 1
        # set precision
        dtype = "float32"  # 8-Byte
        # load torch dataset
        torch_data_train = OASISTorchDataset(directory,
                                             task_type=label_mode,
                                             image_size=image_size,
                                             transforms=transforms,
                                             batch_size=batch_size,
                                             ablation=ablation,
                                             split_name="train",
                                             nch=nch,
                                             seed=seed,
                                             splits=split_ratio_100,
                                             class_names=class_names,
                                             dtype=dtype,
                                             verbose=1)
        self.sample_dataset = torch_data_train
        torch_data_val = OASISTorchDataset(directory,
                                           task_type=label_mode,
                                           image_size=image_size,
                                           transforms=None,
                                           batch_size=batch_size,
                                           ablation=ablation,
                                           split_name="validation",
                                           nch=nch,
                                           seed=seed,
                                           splits=split_ratio_100,
                                           class_names=class_names,
                                           dtype=dtype)
        torch_data_test = OASISTorchDataset(directory,
                                            task_type=label_mode,
                                            image_size=image_size,
                                            transforms=None,
                                            batch_size=batch_size,
                                            ablation=ablation,
                                            split_name="test",
                                            nch=nch,
                                            seed=seed,
                                            splits=split_ratio_100,
                                            class_names=class_names,
                                            dtype=dtype)
        self.classnames = torch_data_train.classnames
        self.num_classes = torch_data_train.num_classes
        self.num_items = len(torch_data_train) + len(torch_data_val) + len(torch_data_test)
        print("Number of patients:", self.num_items)
        print("Class Names:", self.classnames)
        print("Number of classes:", self.num_classes)
        print(f"Minimum MRI waste for {batch_size=} is", self.num_items % batch_size)
        print(f"Minimum MRI slices waste for {batch_size=} is",
              self.num_items % batch_size * torch_data_train.VOLUME_DEPTH)
        self.recommended_batches = [i for i in range(2, self.num_items + 1) if self.num_items % i == 0]
        print(f"0 waste batch recommendations:", self.recommended_batches)
        # convert torch to tf dataset
        train_dataset = self.torch_to_tf(torch_data_train).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = self.torch_to_tf(torch_data_val).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = self.torch_to_tf(torch_data_test).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_dataset.batch(batch_size), val_dataset.batch(batch_size), test_dataset.batch(batch_size)

    def encode_label(self, labels):
        return self.sample_dataset.encode_labels(labels)

    def decode_labels(self, probabilities: np.ndarray, probability_threshold=0.5):
        return self.sample_dataset.decode_labels(probabilities, probability_threshold)
