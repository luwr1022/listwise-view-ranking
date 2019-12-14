from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_align.modules.roi_align import RoIAlignAvg, RoIAlignMax
from model.roi_crop.modules.roi_crop import _RoICrop

import torch
import torch.nn as nn
class RoI_op(object):
    def __init__(self, output_size, feat_size, up_scale, use_cpu = False):
        super(RoI_op, self).__init__()
        self.output_size = output_size
        self.feat_size = feat_size
        self.up_scale = up_scale
        self.use_cpu = use_cpu

        # roi pooling
        self.roi_pool =  _RoIPooling(self.output_size, self.output_size, 1.0 / self.feat_size)

        # roi align
        self.RoIAlignMax = RoIAlignMax(self.output_size, self.output_size, 1.0 / self.feat_size)
        self.RoIAlignAvg = RoIAlignAvg(self.output_size, self.output_size, 1.0 / self.feat_size)

        # roi warp
        self.grid_size = self.output_size * 2
        self.roi_crop = _RoICrop()
        
        # roi refine
        self.up_sample = torch.nn.UpsamplingBilinear2d(scale_factor = up_scale)

    def RoIPooling(self, featrues, rois):
        roi_featrues = self.roi_pool(featrues, rois.view(-1,5))
        return roi_featrues
    
    def RoIAlign(self, featrues, rois, maxp = True):
        if maxp:
            roi_featrues = self.RoIAlignMax(featrues, rois.view(-1,5))
        else:
            roi_featrues = self.RoIAlignAvg(featrues, rois.view(-1,5))
        return roi_featrues
    
    def RoIWarp(self, featrues, rois, maxp = True):
        grid_xy = self._affine_grid_gen(rois.view(-1, 5), featrues.size()[2:], self.grid_size, self.feat_size)
        grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        pooled_features = self.roi_crop(featrues, grid_yx.detach())

        if maxp:
            roi_featrues = torch.nn.functional.max_pool2d(pooled_features, 2, 2)
        else:
            roi_featrues = torch.nn.functional.max_pool2d(pooled_features, 2, 2)
        return roi_featrues
        
    def RoIRefine(self, featrues, rois, maxp = True):
        featrues = self.up_sample(featrues)
        grid_xy = self._affine_grid_gen(rois.view(-1, 5), featrues.size()[2:], self.grid_size, self.feat_size / self.up_scale)
        grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        pooled_features = self.roi_crop(featrues, grid_yx.detach())

        if maxp:
            roi_featrues = torch.nn.functional.max_pool2d(pooled_features, 2, 2)
        else:
            roi_featrues = torch.nn.functional.max_pool2d(pooled_features, 2, 2)
        return roi_featrues

    def _affine_grid_gen(self, rois, input_size, grid_size, feat_size):

        rois = rois.detach()
        x1 = rois[:, 1::4] / feat_size
        y1 = rois[:, 2::4] / feat_size
        x2 = rois[:, 3::4] / feat_size
        y2 = rois[:, 4::4] / feat_size

        height = input_size[0]
        width = input_size[1]

        zero = torch.zeros((rois.size(0), 1))
        if not self.use_cpu:
            zero = zero.cuda()
        theta = torch.cat([\
            (x2 - x1) / (width - 1),
            zero,
            (x1 + x2 - width + 1) / (width - 1),
            zero,
            (y2 - y1) / (height - 1),
            (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

        grid = torch.nn.functional.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

        return grid