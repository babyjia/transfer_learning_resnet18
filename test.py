import os
import time
import numpy as np
from cnn_finetune import make_model
import torch
import torch.nn as nn
import pretrainedmodels
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from options.test_options import TestOptions
from util.visualizer import Visualizer
# define device
# define device


def make_classifier(in_features, num_classes=4):
    classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
    return classifier

def inceptionv3():
    my_inception_v3 = pretrainedmodels.inceptionv3(1000, pretrained='imagenet')
    # my_inception_v3 = torchvision.models.inception_v3(pretrained=True)

    dim_feats = my_inception_v3.last_linear.in_features  # =2048
    nb_classes = 4
    my_inception_v3.last_linear = nn.Linear(dim_feats, nb_classes)
    return my_inception_v3
def save_net_works(opt, which_epoch, net):
    save_filename = '%s_net_%s.pth' % (which_epoch, opt.name)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    save_path = os.path.join(save_dir, save_filename)

    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(net.cpu().state_dict(), save_path)
        net.cuda()
        # net.cuda(opt.gpu_ids[0])

    else:
        torch.save(net.cpu().state_dict(), save_path)


def dpn68():
    my_dpn68 = pretrainedmodels.dpn68(1000, pretrained='imagenet')
    my_dpn68.last_linear = nn.Conv2d(832, 4, kernel_size=(1, 1), stride=(1, 1))
    return my_dpn68


def dpn131():
    my_dpn131 = pretrainedmodels.dpn131(1000, pretrained='imagenet')
    #print(my_dpn131)
    my_dpn131.last_linear = nn.Conv2d(2688, 4, kernel_size=(1, 1), stride=(1, 1))

    return my_dpn131


def senet154():
    my_senet154 = make_model('senet154', num_classes=4, pretrained=True, dropout_p=0.5)
    return my_senet154

class Densenet201(nn.Module):
    def __init__(self, model):
        super(Densenet201, self).__init__()
        self.densenet_layer = model
        # nn.Sequential(*list(model.children()))
        self.fc = nn.Linear(1000, 4)
        # self.cls = nn.Linear(1000,4)

    def forward(self, x):
        x = self.densenet_layer(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        # x = self.cls(x)

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


def pnasnet5large():
    my_pnasnet5large = make_model('pnasnet5large', num_classes=4, pretrained=True, dropout_p=0.5)
    return my_pnasnet5large


def resnext101_32x4d():
    my_resnext = make_model('resnext101_32x4d', num_classes=4, pretrained=True)
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
def load_networks(opt):
    load_filename = '%s_net_%s.pth' % (opt.which_epoch, opt.name)
    load_path = os.path.join(opt.checkpoints_dir, opt.name,load_filename)
    print('loading the model from %s' % load_path)
    state_dic = torch.load(load_path)
    return state_dic

def test_model(opt):
            class_num = np.zeros(4)
            class_wrong = np.zeros(4)
            pred_num = np.zeros(4)
            running_corrects = 0
            my_model.eval()
            for inputs, labels in data_loaders:

                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = my_model(inputs)
                preds = outputs.max(1)[1]
                np_label = labels.cuda().data.cpu().numpy()
                np_preds = preds.cuda().data.cpu().numpy()
                for i in range(4):
                    class_num[i] += (np_label == i).sum()
                    pred_num[i] += (np_preds== i).sum()

                x = (preds == labels)
                wrong = np.where(x == 0)
                if len(wrong)>0:
                    for i in range(len(wrong)):
                        class_wrong[np_label[wrong[i]]]+=1
                        data = data_loaders.dataset
                        #print('take %s wrong as: %s' % (str(np_label[wrong[i]]), str(np_preds[wrong[i]])))
                        #save


                #print(preds)
                running_corrects += torch.sum(preds == labels)
                
                
            epoch_acc = running_corrects.double()/dataset_size    
            print('%s ACC: %.4f'%('test', epoch_acc))
            for i in range(4):
                print('%s ACC: %.4f RECALL: %.4f'%(str(i),
                      (class_num[i] - class_wrong[i])/float(pred_num[i]),
                      (class_num[i] - class_wrong[i]) / float(class_num[i])
                     ))

               
                
          
        


if __name__ =='__main__':
    opt = TestOptions().parse()
    device = torch.device('cuda:{}'.format(opt.gpu_ids) if opt.gpu_ids else 'cpu')
    

    # load data and do data augmention
    path = opt.dataroot
   

    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    #transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    #transforms.RandomResizedCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    if opt.model == 'inceptionv3' :
        transform = transforms.Compose([
                    transforms.Resize(320),
                    transforms.CenterCrop(299),
                    #transforms.Resize(256),
                    #transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # if opt.model == 'senet154':
    #     #or 'pnasnet5large':
    #     transform = transforms.Compose([
    #         # transforms.Resize(320),
    #         # transforms.CenterCrop(299),
    #                 transforms.Resize(256),
    #                 transforms.CenterCrop(224),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #     ])
    kwargs = {'num_workers':4, 'pin_memory':True}

    ###    ImageFolder supose the all the same class files saved in the one  as in a folder
    ###    ImageFolder(root,transform=None,target_transform=None,loader=
    ###    default_loader)


    image_datasets = datasets.ImageFolder(root=os.path.join(path, 'val'), transform = transform)
                
    data_loaders =  DataLoader(image_datasets, batch_size=2, shuffle=True, **kwargs)

    print(image_datasets.class_to_idx)

    
    
  

    dataset_size = len(image_datasets)
    print('#test images \n')
    print(dataset_size)
    module_name = 'test'
    function_name = opt.model
    imp_module = __import__(module_name)
    obj = getattr(imp_module, function_name)

    # chose the model

    my_model = obj()
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


    test_model(opt)



