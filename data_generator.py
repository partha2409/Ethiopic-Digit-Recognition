from torch.utils.data.dataset import Dataset
import numpy as np
import utils
from scipy import ndimage


class DataGenerator(Dataset):

    def __init__(self, hp):
        super(DataGenerator, self).__init__()
        self.hp = hp
        self.train_data_path = hp['train_data_path']
        self.train_images, self.train_labels = utils.load_training_data(self.train_data_path)
        self.total_train_images = self.train_images.shape[0]

    def __getitem__(self, item):
        np.random.seed()

        select_random_img = np.random.randint(self.train_images.shape[0])
        image = self.train_images[select_random_img, :, :]

        image = np.expand_dims(image, 0)

        label = self.train_labels[select_random_img]
        return image, label

    def __len__(self):
        return self.total_train_images


class ValidationDataGenerator(Dataset):

    def __init__(self, hp):
        super(ValidationDataGenerator, self).__init__()
        self.val_data_path = hp['val_data_path']
        self.val_images, self.val_labels = utils.load_validation_data(self.val_data_path)
        self.total_val_images = self.val_images.shape[0]

    def __getitem__(self, item):

        image = self.val_images[item, :, :]
        image = np.expand_dims(image, 0)

        label = self.val_labels[item]

        return image, label

    def __len__(self):
        return self.total_val_images
