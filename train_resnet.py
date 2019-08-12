from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os


# load train img, change line 28 to your own
def reloaddata():
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = 'dataset'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']}
    # if CPU num_workers=0, else num_workers=1
    dset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0)
                  for x in ['train']}
    dset_sizes = len(image_datasets['train'])
    return dset_loaders, dset_sizes


def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=60):
    # batch_size = 64
    train_loss = []
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    # Set model to training mode
    model_ft.train(True)
    for epoch in range(num_epochs):
        dset_loaders, dset_sizes = reloaddata()
        print('Data Size', dset_sizes)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        optimizer = lr_scheduler(optimizer, epoch)

        running_loss = 0.0
        running_corrects = 0
        count = 0

        # Iterate over data.
        for data in dset_loaders['train']:
            # get the inputs
            inputs, labels = data
            # labels must to be LongTensor for CrossEntropyLoss
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            # backward + optimize only if in training phase
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print output
            count += 1
            if count % 30 == 0 or outputs.size()[0] < batch_size:
                print('Epoch:{}: loss:{:.3f}'.format(epoch, loss.item()))
                train_loss.append(loss.item())
            # statistics

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes

        # print('outputs', outputs)
        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model_ft.state_dict()
        if epoch_acc > 0.999:
            break

    # save model
    save_dir = 'dataset/model'
    model_ft.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/best.pth"
    torch.save(model_ft, model_out_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return train_loss


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8**(epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


# Start Training
batch_size = 32
use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

actionType = 8
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, actionType)

# print(model_ft)
criterion = nn.CrossEntropyLoss()
if use_gpu:
    model_ft = model_ft.cuda()
    criterion = criterion.cuda()
    # model_ft = torch.nn.DataParallel(model_ft,device_ids=[0,1,2])
    # model_ft = torch.nn.DataParallel(model_ft,device_ids=[0,1])

optimizer = optim.SGD((model_ft.parameters()), lr=0.01,
                      momentum=0.9, weight_decay=0.0004)

train_loss = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=60)

