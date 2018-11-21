import os
import time
import copy
from cnn_finetune import make_model
import numpy as np
import matplotlib as plt
import pretrainedmodels
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from options.train_options import TrainOptions
from util.visualizer import Visualizer
# define device
# define device
def save_net_works(opt,which_epoch,net):
    save_filename = '%s_net_%s.pth'%(which_epoch,opt.name)
    save_dir = os.path.join(opt.checkpoints_dir,opt.name)
    
    save_path = os.path.join(save_dir,save_filename)
    
    if len(opt.gpu_ids)>0 and torch.cuda.is_available():
        torch.save(net.cpu().state_dict(),save_path)
        net.cuda()
        #net.cuda(opt.gpu_ids[0])

    else:
        torch.save(net.cpu().state_dict(),save_path)
def make_classifier(in_features, num_classes=4):
    classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
    return classifier
def train_model(opt):
        epochs = opt.niter
        visualizer = Visualizer(opt)
        best_model_wts = copy.deepcopy(my_senet154.state_dict())
        best_acc = 0.

        for epoch in range(epochs):
            # in each epoch
            #epoch_start = time.time()
            print('Epoch {}/{}'.format(epoch, epochs-1))
            print('-'*10)
            # iterate on the whole data training set
            loss_dic = {}
           # legend = ['train'+'epoch_loss','train'+'epoch_acc','val'+'epoch_loss','val'+'epoch_acc']
            legend = ['train'+'epoch_acc','val'+'epoch_acc']
            for phase in mode:
                running_loss = 0.
                running_corrects = 0
                if phase == 'train':
                    #exp_lr_scheduler.step()
                    my_senet154.train()
                else :
                    my_senet154.eval()

                # in each epoch iterate over all dataset
                for inputs, labels in data_loaders[phase]:
                    #inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda())
                    inputs = inputs.to(device)
                   
                    labels = labels.to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        # in each iter step
                        # 1. zero the parameter gradients
                        optimizer.zero_grad()

                        # 2. forward 
                        # attention there only need the first one
                        outputs = my_senet154(inputs)
                        # if phase =='train':
                        #     outputs = my_senet154(inputs)[0]
                        
                        

     
                        loss = criterion(outputs,labels)
      
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # statistics
                    preds = outputs.max(1)[1]
                    running_loss += loss.item()*inputs.size(0)
                    running_corrects += torch.sum(preds == labels)
                
                epoch_loss = running_loss/dataset_size[phase]
                epoch_acc = running_corrects.double()/dataset_size[phase]
               # loss_dic[phase+'epoch_loss'] = epoch_loss
                loss_dic[phase+'epoch_acc'] = epoch_acc 
                print('%s Loss: %.4f ACC: %.4f'%(phase, epoch_loss, epoch_acc))
                #
                #print('finished drawing')
                #visualizer.plot_current_losses(epoch,opt,epoch_acc,[phase+'epoch_acc'])
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(my_senet154)
            visualizer.plot_current_losses(epoch,opt,loss_dic,legend)
            if epoch % opt.save_epoch_freq==0:
                save_net_works(opt,epoch,best_model_wts)
        my_senet154.load_state_dict(best_model_wts)
        


if __name__ =='__main__':
    opt = TrainOptions().parse()
    if not torch.cuda.is_available():
        print("cuda is not available")
    device = torch.device('cuda:{}'.format(opt.gpu_ids) if opt.gpu_ids else 'cpu')
    

    # load data and do data augmention
    path = opt.dataroot
    mode = ('train', 'val')


    transform = {
        'train':transforms.Compose([
                    transforms.RandomResizedCrop(224),#just for inception-v3
                    #transforms.RandomResizedCrop(299),
                    transforms.RandomHorizontalFlip(),
                    #以下都是新增加的数据方法
                    transforms.RandomVerticalFlip(), #以0.5的概率垂直翻转
                    transforms.RandomRotation(10), #在（-10， 10）范围内旋转
                    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), #HSV以及对比度变化


                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val':transforms.Compose([
                    #transforms.Resize(320),
                    #transforms.CenterCrop(299),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    kwargs = {'num_workers':4, 'pin_memory':True}

    ###    ImageFolder supose the all the same class files saved in the one  as in a folder
    ###    ImageFolder(root,transform=None,target_transform=None,loader=
    ###    default_loader)


    image_datasets = {x: datasets.ImageFolder(root=os.path.join(path, x), transform = transform[x])
                            for x in mode}

    data_loaders = {x: DataLoader(image_datasets[x], batch_size=opt.batch_size, shuffle=True, **kwargs)
                            for x in mode}
    
    
    class_names = image_datasets['train'].classes

    dataset_size = {x: len(image_datasets[x]) for x in mode}
    print('#training images \n')
    print(dataset_size)


    # define my net and criterion optimizer
    my_senet154 = make_model('senet154',num_classes=4,pretrained=True,dropout_p=0.5)

    my_senet154 = nn.DataParallel(my_senet154).cuda()
    my_senet154 = my_senet154.to(device)



    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(my_senet154.parameters(), lr=opt.lr, momentum=0.9,weight_decay=5e-4)

    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print(my_senet154)



    #import ipdb; ipdb.set_trace()

    # train
    
    # load best model weights
    
     
    train_model(opt)



