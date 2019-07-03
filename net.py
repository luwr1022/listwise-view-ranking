import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

from roi_crop.modules.roi_crop import _RoICrop

def _affine_grid_gen(rois, input_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 8.0
    y1 = rois[:, 2::4] / 8.0
    x2 = rois[:, 3::4] / 8.0
    y2 = rois[:, 4::4] / 8.0

    height = input_size[0]
    width = input_size[1]

    #zero = rois.data.new(rois.size(0), 1).zero_()
    zero = torch.zeros((rois.size(0), 1)).cuda()
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, 14, 14)))

    return grid

class LVRN(nn.Module):    
    def __init__(self, model_path = None):
        super(LVRN, self).__init__()
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.maxpooling = nn.MaxPool2d(2,2)

        self.grid_size = 14
        self.roi_crop = _RoICrop()

        self.features = self.load_model(model_path)
        self.fc_score = nn.Sequential(
            nn.Linear(512 * 49, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 1)
        )

        self.param_init()
        print(self)

    def param_init(self):
        def normal_init(m, mean, stddev, truncated=False):
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        normal_init(self.fc_score[0], 0, 0.01)
        normal_init(self.fc_score[3], 0, 0.01)
        normal_init(self.fc_score[6], 0, 0.01)

    def forward(self, x, roi):
        f = self.features(x)
        f = self.up_sample(f)
        grid_xy = _affine_grid_gen(roi.view(-1, 5), f.size()[2:])   
        grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        pooled_feat = self.roi_crop(f, grid_yx.detach())
        f = self.maxpooling(pooled_feat)
        f = f.view(f.size(0),-1)
        score = self.fc_score(f)
        return score
    
    def load_model(self, model_path = None):
        model = vgg16(pretrained = False)
        if model_path != None:
            model.load_state_dict(torch.load(model_path))
        features = list(model.features)[:30]
        return nn.Sequential(*features)
    
    def load_parameters(self, model_path):
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

        '''
        # conv1_1
        assert model.features[0].weight.size() == self.conv1[0].weight.size()
        assert model.features[0].bias.size() == self.conv1[0].bias.size()
        self.conv1[0].weight.data = model.features[0].weight.data
        self.conv1[0].bias.data = model.features[0].bias.data
        # conv1_2
        assert model.features[2].weight.size() == self.conv1[2].weight.size()
        assert model.features[2].bias.size() == self.conv1[2].bias.size()
        self.conv1[2].weight.data = model.features[2].weight.data
        self.conv1[2].bias.data = model.features[2].bias.data

        # conv2_1
        assert model.features[5].weight.size() == self.conv2[0].weight.size()
        assert model.features[5].bias.size() == self.conv2[0].bias.size()
        self.conv2[0].weight.data = model.features[5].weight.data
        self.conv2[0].bias.data = model.features[5].bias.data
        # conv2_2
        assert model.features[7].weight.size() == self.conv2[2].weight.size()
        assert model.features[7].bias.size() == self.conv2[2].bias.size()
        self.conv2[2].weight.data = model.features[7].weight.data
        self.conv2[2].bias.data = model.features[7].bias.data

        # conv3_1
        assert model.features[10].weight.size() == self.conv3[0].weight.size()
        assert model.features[10].bias.size() == self.conv3[0].bias.size()
        self.conv3[0].weight.data = model.features[10].weight.data
        self.conv3[0].bias.data = model.features[10].bias.data
        # conv3_2
        assert model.features[12].weight.size() == self.conv3[2].weight.size()
        assert model.features[12].bias.size() == self.conv3[2].bias.size()
        self.conv3[2].weight.data = model.features[12].weight.data
        self.conv3[2].bias.data = model.features[12].bias.data
        # conv3_3
        assert model.features[14].weight.size() == self.conv3[4].weight.size()
        assert model.features[14].bias.size() == self.conv3[4].bias.size()
        self.conv3[4].weight.data = model.features[14].weight.data
        self.conv3[4].bias.data = model.features[14].bias.data

        # conv4_1
        assert model.features[17].weight.size() == self.conv4[0].weight.size()
        assert model.features[17].bias.size() == self.conv4[0].bias.size()
        self.conv4[0].weight.data = model.features[17].weight.data
        self.conv4[0].bias.data = model.features[17].bias.data
        # conv4_2
        assert model.features[19].weight.size() == self.conv4[2].weight.size()
        assert model.features[19].bias.size() == self.conv4[2].bias.size()
        self.conv4[2].weight.data = model.features[19].weight.data
        self.conv4[2].bias.data = model.features[19].bias.data
        # conv4_3
        assert model.features[21].weight.size() == self.conv4[4].weight.size()
        assert model.features[21].bias.size() == self.conv4[4].bias.size()
        self.conv4[4].weight.data = model.features[21].weight.data
        self.conv4[4].bias.data = model.features[21].bias.data

        # conv5_1
        assert model.features[24].weight.size() == self.conv5[0].weight.size()
        assert model.features[24].bias.size() == self.conv5[0].bias.size()
        self.conv5[0].weight.data = model.features[24].weight.data
        self.conv5[0].bias.data = model.features[24].bias.data
        # conv5_2
        assert model.features[26].weight.size() == self.conv5[2].weight.size()
        assert model.features[26].bias.size() == self.conv5[2].bias.size()
        self.conv5[2].weight.data = model.features[26].weight.data
        self.conv5[2].bias.data = model.features[26].bias.data
        # conv5_3
        assert model.features[28].weight.size() == self.conv5[4].weight.size()
        assert model.features[28].bias.size() == self.conv5[4].bias.size()
        self.conv5[4].weight.data = model.features[28].weight.data
        self.conv5[4].bias.data = model.features[28].bias.data

        #assert model.up_features.weight.size() == self.up_features.weight.size()
        #self.up_features.weight.data = model.up_features.weight.data

        assert model.fc_score[0].weight.size() == self.fc_score[0].weight.size()
        assert model.fc_score[0].bias.size() == self.fc_score[0].bias.size()
        self.fc_score[0].weight.data = model.fc_score[0].weight.data
        self.fc_score[0].bias.data = model.fc_score[0].bias.data
        
        assert model.fc_score[3].weight.size() == self.fc_score[3].weight.size()
        assert model.fc_score[3].bias.size() == self.fc_score[3].bias.size()
        self.fc_score[3].weight.data = model.fc_score[3].weight.data
        self.fc_score[3].bias.data = model.fc_score[3].bias.data
        
        assert model.fc_score[6].weight.size() == self.fc_score[6].weight.size()
        assert model.fc_score[6].bias.size() == self.fc_score[6].bias.size()
        self.fc_score[6].weight.data = model.fc_score[6].weight.data
        self.fc_score[6].bias.data = model.fc_score[6].bias.data

        #for p in self.fc_score.parameters():
        #    p.requires_grad = True
        '''


