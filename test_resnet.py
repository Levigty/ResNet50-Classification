from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import numpy as np
import torch.nn.functional as FUN
import os
from scipy import io


# load test img
def reloaddata():
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            #transforms.Scale(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}
    dset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=False, num_workers=0)
                  for x in ['test']}
    dset_sizes = len(image_datasets['test'])
    return dset_loaders, dset_sizes


def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_sizes = reloaddata()
    # Iterate over data.
    for data in dset_loaders['test']:
        # get the inputs
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        # GPU
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        # CPU
        inputs, labels = Variable(inputs), Variable(labels)
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        # print('dset_sizes:',dset_sizes)
        print('Num:', cont)
        cont += 1

    print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / dset_sizes,
                                            running_corrects.double() / dset_sizes))

    return FUN.softmax(Variable(outPre)).data.numpy(), outLabel.numpy()


# Start Testing
data_dir = 'dataset'
save_dir = 'dataset/model'
modelft_file = save_dir + "/best.pth"
batch_size = 32

# GPU时
# model_ft = torch.load(modelft_file).cuda()
# criterion = nn.CrossEntropyLoss().cuda()

# CPU时
model_ft = torch.load(modelft_file, map_location='cpu')
criterion = nn.CrossEntropyLoss()

outPre, outLabel = test_model(model_ft, criterion)

# Save result
np.save(save_dir + '/Pre', outPre)
np.save(save_dir + '/Label', outLabel)

# Change the result and scores to .mat
mat = np.load(save_dir + '/Pre.npy')
io.savemat(save_dir + '/Pre.mat', {'gene_features': mat})

label = np.load(save_dir + '/Label.npy')
io.savemat(save_dir + '/Label.mat', {'labels': label})

