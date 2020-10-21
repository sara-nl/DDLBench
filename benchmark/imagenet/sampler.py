import glob
from skimage import io
import os
import shutil
import time
from tqdm import tqdm
import argparse

# Imagenet_dir /nfs/managed_datasets/imagenet-full/
class ImageNetDataset:
    def __init__(self, imagenet_dir, train, scratch_dir, copy_files: bool, num_classes=1):
        super(ImageNetDataset, self).__init__()
        train = train
        train_folder = os.path.join(imagenet_dir, 'train')
        test_folder = os.path.join(imagenet_dir, 'test')

        classes_train = set(d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d)))
        classes_test = set(d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d)))
        classes_train = classes_test = classes_train.intersection(classes_test)

        classes_train = sorted(list(classes_train))[:num_classes]
        classes_test = sorted(list(classes_test))[:num_classes]

        assert len(classes_train) == len(classes_test) == num_classes

        self.label_to_ix = {label: i for i, label in enumerate(classes_train)}  # {n02034234: 1, n23425325: 2, ...}
        self.ix_to_label = {i: label for label, i in self.label_to_ix.items()}

        train_examples = []
        self.train_labels = []

        for label in classes_train:
            for f in glob.glob(os.path.join(train_folder, label) + '/*.JPEG'):
                self.train_labels.append(self.label_to_ix[label])
                train_examples.append(f)

        test_examples = [] #['/nfs/managed-datasets/test/n013203402/01.jpg', ...]
        self.test_labels = []  # [495, ...]

        for label in classes_test:
            for f in glob.glob(os.path.join(test_folder, label) + '/*.JPEG'):
                self.test_labels.append(self.label_to_ix[label])
                test_examples.append(f)

        scratch_dir = os.path.normpath(scratch_dir)

        self.scratch_files_train = []
        self.scratch_files_test = []

        if train:
            print("Copying train files to scratch...")
            for f in tqdm(sorted(train_examples)):
                if copy_files:
                    directory = os.path.normpath(scratch_dir + f.rsplit('/', maxsplit=1)[0])
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    shutil.copy(f, os.path.normpath(scratch_dir + f))
                self.scratch_files_train.append(os.path.normpath(scratch_dir + f))
            if copy_files:
                print("All Files Copied")

            while not all(os.path.exists(f) for f in self.scratch_files_train):
                print(f"Waiting... {sum(os.path.exists(f) for f in self.scratch_files_train)} / {len(self.scratch_files_train)}")
                time.sleep(1)

            assert all(os.path.isfile(f) for f in self.scratch_files_train)
            print(f"Length of train dataset: {len(self.scratch_files_train)}")
            print(f"Length of train labels: {len(self.train_labels)}")
            assert len(self.scratch_files_train) == len(self.train_labels)
            test_image = io.imread(train_examples[0])

        if not train:
            print("Copying test files to scratch...")
            for f in tqdm(sorted(test_examples)):
                if copy_files:
                    directory = os.path.normpath(scratch_dir + f.rsplit('/', maxsplit=1)[0])
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    shutil.copy(f, os.path.normpath(scratch_dir + f))
                self.scratch_files_test.append(os.path.normpath(scratch_dir + f))

            while not all(os.path.exists(f) for f in self.scratch_files_test):
                print(f"Waiting... {sum(os.path.exists(f) for f in self.scratch_files_test)} / {len(self.scratch_files_test)}")
                time.sleep(1)

            assert all(os.path.isfile(f) for f in self.scratch_files_test)

            print(f"Length of test dataset: {len(self.scratch_files_test)}")
            print(f"Length of test labels: {len(self.test_labels)}")
            assert len(self.scratch_files_test) == len(self.test_labels)
            test_image = io.imread(test_examples[0])

        self.shape = test_image.shape
        self.dtype = test_image.dtype

        self.is_train = train

        del test_image

    def __len__(self):
        if self.is_train:
            return len(self.scratch_files_train)
        else:
            return len(self.scratch_files_test)

    def get_data(self):
        if self.is_train:
            return self.scratch_files_train, self.train_labels
        else:
            return self.scratch_files_test, self.test_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Imagement sampler',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-dir', default=os.environ['TMPDIR'] + '/train',
                        help='path to training data')
    parser.add_argument('--val-dir', default=os.environ['TMPDIR'] + '/val',
                        help='path to validation data')
    parser.add_argument('--classes', type=int, default=1,
                    help='Number of classes to sample. ~ 1000 imgs per class.')
    args = parser.parse_args()

    print('Start copying')
    _ = ImageNetDataset('/nfs/managed_datasets/imagenet-full/', train=True, scratch_dir=args.train_dir, copy_files=True, num_classes=args.classes)
    _ = ImageNetDataset('/nfs/managed_datasets/imagenet-full/', train=False, scratch_dir=args.val_dir, copy_files=True, num_classes=args.classes)
    print("Done copying")
