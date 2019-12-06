import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet50, alexnet, squeezenet1_0
from model import RoI_operation

class LVRN(nn.Module):    
    def __init__(self, model_path = None, fc1 = 1024, fc2 = 512):
        super(LVRN, self).__init__()
        self.features = self.load_vgg16(model_path)
        
        self.fc_score = nn.Sequential(
            nn.Linear(512 * 49, fc1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fc1, fc2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fc2, 1)
        )

        self.RoI_op   = RoI_operation.RoI_op(7, 16, 2)
        self.param_init()
        print(self)
        print("create LVRN successfully!")

    def param_init(self):
        def normal_init(m, mean, stddev, truncated = False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        normal_init(self.fc_score[0], 0, 0.01)
        normal_init(self.fc_score[3], 0, 0.01)
        normal_init(self.fc_score[6], 0, 0.01)

    def forward(self, x, rois):
        f = self.features(x)
        f = self.RoI_op.RoIRefine(f, rois)
        f = f.view(f.size(0), -1)
        score = self.fc_score(f)
        return score
    
    def load_vgg16(self, model_path = None):
        model = vgg16(pretrained = False)
        if model_path != None:
            model.load_state_dict(torch.load(model_path))
        features = list(model.features)[:30]
        return nn.Sequential(*features)
    
    def load_resnet50(self, model_path = None):
        model = resnet50(pretrained = False)
        if model_path != None:
            model.load_state_dict(torch.load(model_path))
        features = [model.conv1, model.bn1, model.relu, model.maxpool, 
                    model.layer1, model.layer2, model.layer3, model.layer4]
        return nn.Sequential(*features)

    def load_alexnet(self, model_path = None):
        model = alexnet(pretrained = False)
        if model_path != None:
            model.load_state_dict(torch.load(model_path))
        features = list(model.features)[:12]
        return nn.Sequential(*features)
        
    def load_squeezenet1_0(self, model_path = None):
        model = squeezenet1_0(pretrained = False)
        if model_path != None:
            model.load_state_dict(torch.load(model_path))
        features = list(model.features)[:12]
        return nn.Sequential(*features)

    def load_pretrain_parameters(self, model_path):
        checkpoint = torch.load(model_path)

        for module_name in checkpoint['model']:
            p = module_name.split('.')
            if len(p) == 3:
                if p[0] == 'features':
                    if p[2] == 'weight':
                        self.features[int(p[1])].weight.data = checkpoint['model'][module_name]
                    else:
                        self.features[int(p[1])].bias.data = checkpoint['model'][module_name]
                else:
                    if p[2] == 'weight':
                        self.fc_score[int(p[1])].weight.data = checkpoint['model'][module_name]
                    else:
                        self.fc_score[int(p[1])].bias.data = checkpoint['model'][module_name]

if __name__=="__main__":
    model = LVRN(None, 1024, 512)
    model = model.cuda()
    roi = torch.tensor([[0,0,0,128,128],[0,64,64,223,223]]).float().cuda()
    img = torch.rand((1,3,224,224)).float().cuda()
    score = model(img, roi)
    print(score)
