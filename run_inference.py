import torch
from model import ConvNet
import utils
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import csv


class TestDataGenerator(Dataset):

    def __init__(self, test_data_path):
        super(TestDataGenerator, self).__init__()
        self.test_data_path = test_data_path
        self.test_images = utils.load_test_data(self.test_data_path)
        self.total_test_images = self.test_images.shape[0]

    def __getitem__(self, item):

        image = self.test_images[item, :, :]
        image = np.expand_dims(image, 0)
        return image

    def __len__(self):
        return self.total_test_images


def run_inference(model_dir=None, model_path=None, test_data_path=None):
    hp_file = model_dir + 'hp.pkl'
    f = open(hp_file, "rb")
    hp = pickle.load(f)

    model = ConvNet().to(device)
    ckpt = torch.load(model_dir+model_path, map_location=device)
    model.load_state_dict(ckpt['net'])

    batch_size = 100
    dataset = TestDataGenerator(test_data_path)
    iterator = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=hp['num_workers'], pin_memory=True,
                          shuffle=False, drop_last=True)

    prediction = np.zeros(10000)

    with torch.no_grad():
        for i, img in enumerate(iterator):
            print(i)
            img = img.to(device, dtype=torch.float)
            logits = model(img)
            logits = logits.detach().cpu().numpy()
            pred_label = np.argmax(logits, axis=1)
            pred_label = pred_label + 1    # class numbers 1 - 10, predicted labels 0 - 9
            prediction[i*batch_size: i*batch_size + batch_size] = pred_label

    test_image_names = os.listdir(test_data_path)
    for i, name in enumerate(test_image_names):
        test_image_names[i] = test_image_names[i][0:5]

    with open(model_dir + 'predictions.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Id', 'Category'])
        for i in range(len(test_image_names)):
            writer.writerow([test_image_names[i], int(prediction[i])])


if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_dir = "checkpoints/ResNet_20201108_222637/"
    model_path = "pretrained_model.pth"
    test_data_path = "tau-ethiopic-digit-recognition/test/"

    run_inference(model_dir, model_path, test_data_path)