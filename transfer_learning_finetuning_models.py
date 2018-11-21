import os
import time
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pretrainedmodels
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from cnn_finetune import make_model
# define device
# define device
def load_networks(opt):
    load_filename = '%s_net_%s.pth' % (opt.which_epoch, opt.name)
    load_path = os.path.join(opt.checkpoints_dir, opt.name,load_filename)
    print('loading the model from %s' % load_path)
    state_dic = torch.load(load_path)
    return state_dic

def inceptionv3():
    my_inception_v3 = pretrainedmodels.inceptionv3(1000, pretrained='imagenet')
    # my_inception_v3 = torchvision.models.inception_v3(pretrained=True)

    dim_feats = my_inception_v3.last_linear.in_features  # =2048
    nb_classes = 4
    my_inception_v3.last_linear = nn.Linear(dim_feats, nb_classes)
    return my_inception_v3
def make_classifier(in_features, num_classes=4):
    classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
    return classifier
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
class Densenet201(nn.Module):
    def __init__(self,model):
        super(Densenet201,self).__init__()
        self.densenet_layer = model
        #nn.Sequential(*list(model.children()))
        self.fc = nn.Linear(1000, 4)
        #self.cls = nn.Linear(1000,4)
    def forward(self,x):
        x = self.densenet_layer(x)
       # x = x.view(x.size(0), -1)
        x = self.fc(x)

       #x = self.cls(x)

        return x
def densenet201():
    densenet201 = torchvision.models.densenet201(pretrained=True)
    my_model = Densenet201(densenet201)
    pretrained_dict = densenet201.state_dict()
    model_dict = my_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    my_model.load_state_dict(model_dict)
    return my_model
def dpn68():
    my_dpn68 = pretrainedmodels.dpn68(1000, pretrained='imagenet')
    my_dpn68.last_linear = nn.Conv2d(832, 4, kernel_size=(1, 1), stride=(1, 1))
    return my_dpn68


def dpn131():
    my_dpn131 = pretrainedmodels.dpn131(1000, pretrained='imagenet')
    print(my_dpn131)
    my_dpn131.last_linear = nn.Conv2d(2688, 4, kernel_size=(1, 1), stride=(1, 1))

    return my_dpn131


def pnasnet5large():
    my_pnasnet5large = make_model('pnasnet5large', num_classes=4, pretrained=True, dropout_p=0.5)
    return my_pnasnet5large
def resnext101_32x4d():
    my_resnext = make_model('resnext101_32x4d', num_classes=4, pretrained=True)
    return my_resnext

def resnext101_64x4d():
    my_resnext = make_model('resnext101_64x4d', num_classes=4, pretrained=True)
    return my_resnext


def resnet101():
    my_resnet101 = torchvision.models.resnet101(pretrained=True)
    num_features = my_resnet101.fc.in_features
    my_resnet101.fc = nn.Linear(num_features, 4)

    return my_resnet101
def resnet18():
    my_resnet18 = torchvision.models.resnet18(pretrained=True)

    num_features = my_resnet18.fc.in_features

    my_resnet18.fc = nn.Linear(512, 4)
    return my_resnet18
def get_number_in_class(root,class_name):
    class_num = len(class_name)
    count_class = np.zeros(class_num)
    for i in range(class_num):
        imgs_dir = os.path.join(root,class_name[i])
        imgs = os.listdir(imgs_dir)
        count_class[i] = len(imgs)
    return count_class

def class_weight(class_name):
    num_class = get_number_in_class(os.path.join(path, 'train'), class_name)
    class_weight = np.zeros(len(num_class))
    sum_weight = 0.0
    for i in range(len(num_class)):
        # class_count[i] = (labels_np==class_name[i]).sum()
        sum_weight += ((1.0) / (num_class[i] + 0.001))
    for i in range(len(num_class)):
        class_weight[i] = ((1.0) / (num_class[i] + 0.001)) / sum_weight
    class_weight = torch.Tensor(class_weight)
    return class_weight


def senet154():
    my_senet154 = make_model('senet154', num_classes=4, pretrained=True, dropout_p=0.5)
    return my_senet154


def train_model(opt):
        epochs = opt.niter
        visualizer = Visualizer(opt)
        best_model_wts = copy.deepcopy(my_model.state_dict())
        best_acc = 0.

        for epoch in range(epochs):
            # in each epoch
            epoch_start = time.time()
            print('Epoch {}/{}'.format(epoch, epochs-1))
            print('-'*10)
            # iterate on the whole data training set
            loss_dic = {}

            legend = ['train'+'epoch_acc','val'+'epoch_acc']

            for phase in mode:
                running_loss = 0.
                running_corrects = 0
                if phase == 'train':
                    #exp_lr_scheduler.step()
                    my_model.train()
                else :
                    my_model.eval()

                # in each epoch iterate over all dataset

                for inputs, labels in data_loaders[phase]:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.set_grad_enabled(phase == 'train'):                   
                        inputs = inputs.to(device)
                        # in each iter step
                        # 1. zero the parameter gradients
                        optimizer.zero_grad()
                    
                        # 2. forward

                        outputs = my_model(inputs)
                        if opt.model == 'inceptionv3' and phase == 'train':

                            outputs = my_model(inputs)[0]

                        # 3. compute loss and backward and update parameters
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # statistics
                    preds = outputs.max(1)[1]
 
                    running_loss += loss.item()*inputs.size(0)
                    running_corrects += torch.sum(preds == labels)
                
                epoch_loss = running_loss/dataset_size[phase]
                epoch_acc = running_corrects.double()/dataset_size[phase]
                loss_dic[phase+'epoch_acc'] = epoch_acc 
                print('%s Loss: %.4f ACC: %.4f'%(phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(my_model)
            visualizer.plot_current_losses(epoch,opt,loss_dic,legend)
            if epoch % opt.save_epoch_freq==0:
                save_net_works(opt,epoch,best_model_wts)
        


if __name__ =='__main__':
    opt = TrainOptions().parse()
    device = torch.device('cuda:{}'.format(opt.gpu_ids) if opt.gpu_ids else 'cpu')

    # load data and do data augmention
    path = opt.dataroot
    mode = ('train', 'val')

    transform = {
        'train':transforms.Compose([
                    # transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    #以下都是新增加的数据方法

                    transforms.RandomVerticalFlip(), # 以0.5的概率垂直翻转
                    transforms.RandomRotation(10), # 在（-10， 10）范围内旋转
                    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), #HSV以及对比度变化

                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val':transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    #transforms.RandomResizedCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }
    print("test for harmony")
    if opt.model =='inceptionv3':
        print('transform for inceptionv3')
        transform = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(224),just for inception-v3
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                # 以下都是新增加的数据方法
                transforms.RandomVerticalFlip(),  # 以0.5的概率垂直翻转
                transforms.RandomRotation(10),  # 在（-10， 10）范围内旋转
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),  # HSV以及对比度变化

                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'val': transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        }

    kwargs = {'num_workers':4, 'pin_memory':True}



    image_datasets = {x: datasets.ImageFolder(root=os.path.join(path, x), transform = transform[x])
                            for x in mode}

    data_loaders = {x: DataLoader(image_datasets[x], batch_size=opt.batch_size, shuffle=True, **kwargs)
                            for x in mode}
    
    
  
    dataset_size = {x: len(image_datasets[x]) for x in mode}
    print('#training images \n')
    print(dataset_size)
    module_name = 'transfer_learning_finetuning_models'
    function_name =  opt.model
    imp_module = __import__(module_name)
    obj = getattr(imp_module,  function_name)

    #chose the model

    my_model = obj()
    if  opt.continue_train:
        pretrained_net = load_networks(opt)

        pre_dict = my_model.state_dict()
        if opt.model == 'senet154':
            pretrained_net2 = {k.replace('module._', '_'): v for k, v in pretrained_net.items()}
        else:
            pretrained_net2 = {k.replace('module.', ''): v for k, v in pretrained_net.items()}
        pretrained_dict = {k: v for k, v in pretrained_net2.items() if k in pre_dict}

        my_model.load_state_dict(pretrained_dict)
    my_model = nn.DataParallel(my_model).cuda()

    my_model = my_model.to(device)
    criterion = nn.CrossEntropyLoss()
    if opt.class_weight:
        class_name = image_datasets['train'].classes
        class_weight = class_weight(class_name)
        class_weight = class_weight.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.SGD(my_model.parameters(), lr=opt.lr, momentum=0.9,weight_decay=5e-4)
    print(my_model)     
    train_model(opt)



