import torch
import numpy as np
from torch.utils.data import DataLoader
from model import ConvNet, CrossEntropyLoss
from data_generator import DataGenerator, ValidationDataGenerator
from config import hp
import os
import time
import matplotlib.pyplot as plt
import pickle
import utils


def train(pre_trained=None):

    # create folder to save models and loss graphs

    reference = hp['net_type'] + str(time.strftime("_%Y%m%d_%H%M%S"))
    checkpoints_folder = hp["output_dir"] + '/checkpoints/' + reference
    os.makedirs(checkpoints_folder, exist_ok=True)

    # save hyper parameter settings
    pickle_file_location = checkpoints_folder + "/hp.pkl"
    pickle_file = open(pickle_file_location, "wb")
    pickle.dump(hp, pickle_file)
    pickle_file.close()

    # create data iterator
    train_data_set = DataGenerator(hp)
    iterator = DataLoader(dataset=train_data_set, batch_size=hp['batch_size'], num_workers=hp['num_workers'], pin_memory=True, shuffle=False, drop_last=True)

    val_set = ValidationDataGenerator(hp)
    val_set_iterator = DataLoader(dataset=val_set, batch_size=50, num_workers=hp['num_workers'], pin_memory=True, shuffle=False, drop_last=True)
    # create model and loss

    model = ConvNet().to(device)
    loss = CrossEntropyLoss().to(device)

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hp['learning_rate'])

    start_epoch = 0
    # load pre trained model

    if pre_trained is not None:
        ckpt = torch.load(pre_trained)
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1

    # init loss arrays
    classification_loss = np.zeros(hp['num_epochs'])
    train_accuracy = np.zeros(hp['num_epochs'])
    val_accuracy = np.zeros(hp['num_epochs'])

    # training loop
    for epoch in range (start_epoch, hp['num_epochs']):
        c_loss = 0
        acc = 0
        for i, (img, label) in enumerate(iterator):
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)

            optimizer.zero_grad()
            logits = model(img)
            l = loss(logits, label.long())
            l.backward()
            optimizer.step()

            c_loss += l.item()
            # calc accuracy
            logits = logits.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            acc += utils.classification_accuracy(logits, label)
            print("epoch = {}, Training_sample={}, classification loss ={}".format(epoch, i, l.item()))

        # average loss per epoch
        classification_loss[epoch] = c_loss/(i+1)
        # average accuracy per epoch
        train_accuracy[epoch] = acc/(i+1)

        print("epoch = {}, average classification loss ={}".format(epoch, classification_loss[epoch]))
        print("epoch = {}, Training accuracy ={}".format(epoch, train_accuracy[epoch]))

        with torch.no_grad():
            val_acc = 0
            for i, (img, label) in enumerate(val_set_iterator):
                img = img.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)
                logits = model(img)
                # calc accuracy
                logits = logits.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                val_acc += utils.classification_accuracy(logits, label)

        val_accuracy[epoch] = val_acc/(i+1)
        print("epoch = {},  Validation set accuracy ={}".format(epoch, val_accuracy[epoch]))

        # plot accuracy curves and save model
        plt.plot(range(1, len(train_accuracy)+1), train_accuracy, 'b-', label=" Train Accuracy")
        plt.plot(range(1, len(val_accuracy)+1), val_accuracy, 'r-', label="Validation Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='best')
        plt.savefig(checkpoints_folder + "/accuracy.jpeg", bbox_inches="tight")
        plt.clf()

        net_save = {'net': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}
        torch.save(net_save, checkpoints_folder + "/convnet_ethiopian_mnist_epoch{}.pth".format(epoch))


if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    pre_trained_model_path = None    # provide path to .pth to continue training
    train(pre_trained=pre_trained_model_path)

