import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader
import time
from imgaug import augmenters

# pipline
tfs = torchvision.transforms.Compose([
    # augmenters.Sequential([
    #     augmenters.Affine(rotate=(-10, 10), shear=(-5, 5), mode="reflect"),
    #     augmenters.CropAndPad(percent=(-0.1, 0.1)),
    #     augmenters.AdditiveGaussianNoise(scale=(0, 0.05)),
    #     augmenters.ContrastNormalization(alpha=(0.8, 1.2)),
    # ]).augment_image,
    torchvision.transforms.ToTensor()
])

# training device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare dataset
train_data = torchvision.datasets.CIFAR10("./Dataset", train=True, transform=tfs,
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./Dataset", train=False, transform=tfs,
                                         download=True)

# length of train dataset
test_data_size = len(test_data)
print("train_data_size: {}".format(test_data_size))
train_data_size = len(train_data)
print("train_data_size: {}".format(train_data_size))

# dataloader
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# create model
net = torchvision.models.vgg16(pretrained=True)
net = net.to(device)

# loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# optimizer
# 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, verbose=True,
#                                                        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
#                                                        eps=1e-08)

# set parameters
# training times
total_train_step = 0
# testing times
total_test_step = 0
# training epoch
epoch = 50

# tensorboard
writer = SummaryWriter("./logs_train")
start_time = time.time()

# find max accuracy
best_accuracy = 0.0
best_epoch = -1

num = 0
for i in range(epoch):
    print("-----------epoch: {}-----------".format(i + 1))

    # train
    net.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)
        # optimizer optimize model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("Time: {}".format(end_time - start_time))
            print("Total_train_step: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("Train_loss", loss.item(), total_train_step)

    # TEST
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets.to(device)).sum()
            total_accuracy += accuracy

    print("Test_Loss: {}".format(total_test_loss))
    print("Test_Accuracy: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("Test_loss", total_test_loss, total_test_step)
    writer.add_scalar("Test_Accuracy", total_accuracy / test_data_size, total_test_step)
    if best_accuracy < total_accuracy / test_data_size:
        best_accuracy = total_accuracy / test_data_size
        best_epoch = i
        torch.save(net, "./model_training/model_{}.pth".format(num))
        print("Saved model_{}.pth".format(num))
        num += 1
    total_test_step += 1

writer.close()
print(f"Best_Accuracy: {best_accuracy} , in Epoch{best_epoch} ")
