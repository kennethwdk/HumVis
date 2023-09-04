import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

from .smpl import SMPL_layer
from .utils import rot6d_to_rotmat, rotmat_to_axis_angle, _sigmoid

def build_iia_module(cfg):
    return IIA(cfg)

def build_gfd_module(cfg):
    return GFD(cfg)

class IIA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.in_channels = cfg.MODEL.IIA.IN_CHANNELS
        self.out_channels = cfg.MODEL.IIA.OUT_CHANNELS
        assert self.out_channels == self.num_keypoints + 1
        self.num_part = 15

        self.heatmap_head = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.mask_head = nn.Conv2d(self.in_channels, 1, 1, 1, 0)
        self.part_head = nn.Conv2d(self.in_channels, self.num_part, 1, 1, 0)
        # smpl mesh part
        self.kpts_head = nn.Conv2d(self.in_channels, self.num_keypoints*2, 1, 1, 0)
        param_out_dim = 24*6 + 10 + 3
        self.mesh_head = nn.Conv2d(self.in_channels, param_out_dim, 1, 1, 0)
        self.smpl_joint_select = [1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]
        self.coco_joint_select = [11, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10]

        # smpl layer
        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )
        self.focal = cfg.MODEL.get('FOCAL', 5000)
        self.camera_3d_size = cfg.MODEL.get('CAMERA_3D_SIZE', 2.5)

        # inference
        self.flip_test = cfg.TEST.FLIP_TEST
        self.max_proposals = cfg.TEST.MAX_PROPOSALS
        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2
    
    def get_camera_trans(self, cam_param, h):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.pow(1.1, cam_param[:, 2])
        # gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(self.focal*self.focal*self.camera_3d_size*self.camera_3d_size/(h*h))]).to(cam_param.device).view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def forward(self, features, batch_inputs=None, input_scale=1.0):
        pred_multi_heatmap = _sigmoid(self.heatmap_head(features))
        pred_multi_mask = self.mask_head(features).sigmoid()
        pred_multi_part = self.part_head(features)

        pred_offsets = self.kpts_head(features)
        pred_smplparam = self.mesh_head(features)
        locations = self.locations(pred_smplparam)[None, None]
        bs, c, h, w = features.shape
        pred_keypoints = locations - pred_offsets.reshape(bs, self.num_keypoints, 2, h, w)
        pred_keypoints = pred_keypoints.reshape(bs, self.num_keypoints*2, h, w)

        instances = {}
        H, W = pred_multi_heatmap.size()[2:]
        if self.flip_test:
            center_heatmap = pred_multi_heatmap[:, -1, :, :].mean(dim=0, keepdim=True)
        else:
            center_heatmap = pred_multi_heatmap[:, -1, :, :]

        center_pool = F.avg_pool2d(center_heatmap, 3, 1, 1)
        center_heatmap = (center_heatmap + center_pool) / 2.0
        maxm = self.hierarchical_pool(center_heatmap)
        maxm = torch.eq(maxm, center_heatmap).float()
        center_heatmap = center_heatmap * maxm
        scores = center_heatmap.view(-1)
        scores, pos_ind = scores.topk(self.max_proposals, dim=0)
        select_ind = (scores > (self.keypoint_thre)).nonzero()
        if len(select_ind) > 0:
            scores = scores[select_ind].squeeze(1)
            pos_ind = pos_ind[select_ind].squeeze(1)
            x = pos_ind % W
            y = (pos_ind / W).long()
            instance_coord = torch.stack((y, x), dim=1)
            instance_param = self._sample_feats(features[0], instance_coord)
            instance_imgid = torch.zeros(instance_coord.size(0), dtype=torch.long).to(features.device)
            if self.flip_test:
                instance_param_flip = self._sample_feats(features[1], instance_coord)
                instance_imgid_flip = torch.ones(instance_coord.size(0), dtype=torch.long).to(features.device)
                instance_coord = torch.cat((instance_coord, instance_coord), dim=0)
                instance_param = torch.cat((instance_param, instance_param_flip), dim=0)
                instance_imgid = torch.cat((instance_imgid, instance_imgid_flip), dim=0)
            
            pred_smplparam = pred_smplparam.permute(0, 2, 3, 1).reshape(bs*h*w, -1)
            pred_param = pred_smplparam[pos_ind]
            pred_pose_6d = pred_param[:, :24*6]
            pred_shape = pred_param[:, 24*6:24*6+10]
            pred_cam = pred_param[:, -3:]
            pred_pose_rotmat = rot6d_to_rotmat(pred_pose_6d.reshape(-1, 6)).reshape(-1, 24*9)
            pred_pose_axis = rotmat_to_axis_angle(pred_pose_rotmat.reshape(-1, 3, 3)).reshape(-1, 24*3)
            pred_outputs = self.smpl(pred_pose_axis.contiguous(), pred_shape)
            cam_trans = self.get_camera_trans(pred_cam, 512*input_scale)
            smpl_mesh_cam = pred_outputs.vertices + cam_trans[:, None, :]
            location = locations[0].permute(0, 2, 3, 1).reshape(bs*h*w, -1)[pos_ind].reshape(-1, 2) * 4 * input_scale

            instances['instance_coord'] = instance_coord
            instances['instance_imgid'] = instance_imgid
            instances['instance_param'] = instance_param
            instances['instance_score'] = scores
            instances['mesh'] = smpl_mesh_cam
            instances['location'] = location

        return instances
    
    def _sample_feats(self, features, pos_ind):
        feats = features[:, pos_ind[:, 0], pos_ind[:, 1]]
        return feats.permute(1, 0)

    def hierarchical_pool(self, heatmap):
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        if map_size > self.pool_thre1:
            maxm = F.max_pool2d(heatmap, 7, 1, 3)
        elif map_size > self.pool_thre2:
            maxm = F.max_pool2d(heatmap, 5, 1, 2)
        else:
            maxm = F.max_pool2d(heatmap, 3, 1, 1)
        return maxm

    @torch.no_grad()
    def locations(self, features):
        h, w = features.size()[-2:]
        device = features.device
        shifts_x = torch.arange(0, w, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)        
        locations = locations.reshape(h, w, 2).permute(2, 0, 1)
        return locations

class GFD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.in_channels = cfg.MODEL.GFD.IN_CHANNELS
        self.channels = cfg.MODEL.GFD.CHANNELS
        self.out_channels = cfg.MODEL.GFD.OUT_CHANNELS
        assert self.out_channels == self.num_keypoints
        self.prior_prob = cfg.MODEL.BIAS_PROB
        self.num_part = 15

        self.conv_down = nn.Conv2d(self.in_channels, self.channels, 1, 1, 0)
        self.c_attn = ChannelAtten(self.in_channels, self.channels)
        self.s_attn = SpatialAtten(self.in_channels, self.channels)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(self.channels*2, self.channels, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(self.channels, self.out_channels, 1, 1, 0)
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.channels*2, self.channels, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(self.channels, 1, 1, 1, 0)
        )
        self.part_head = nn.Sequential(
            nn.Conv2d(self.channels*2, self.channels, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(self.channels, self.num_part, 1, 1, 0)
        )

    def forward(self, features, instances):
        global_features = self.conv_down(features)
        instance_features = global_features[instances['instance_imgid']]
        instance_params = instances['instance_param']
        c_instance_feats = self.c_attn(instance_features, instance_params)
        s_instance_feats = self.s_attn(instance_features, instance_params, instances['instance_coord'])
        cond_instance_feats = torch.cat((c_instance_feats, s_instance_feats), dim=1)
        
        pred_instance_heatmaps = _sigmoid(self.heatmap_head(cond_instance_feats))
        pred_instance_masks = self.mask_head(cond_instance_feats).sigmoid()
        pred_instance_parts = self.part_head(cond_instance_feats)

        return pred_instance_heatmaps, pred_instance_masks, F.softmax(pred_instance_parts, dim=1)

class ChannelAtten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAtten, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels)

    def forward(self, global_features, instance_params):
        B, C, H, W = global_features.size()
        instance_params = self.atn(instance_params).reshape(B, C, 1, 1)
        return global_features * instance_params.expand_as(global_features)

class SpatialAtten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAtten, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels)
        self.feat_stride = 4
        conv_in = 3
        self.conv = nn.Conv2d(conv_in, 1, 5, 1, 2)

    def forward(self, global_features, instance_params, instance_inds):
        B, C, H, W = global_features.size()
        instance_params = self.atn(instance_params).reshape(B, C, 1, 1)
        feats = global_features * instance_params.expand_as(global_features)
        fsum = torch.sum(feats, dim=1, keepdim=True)
        input_feats = fsum
        locations = compute_locations(global_features.size(2), global_features.size(3), stride=1, device=global_features.device)
        n_inst = instance_inds.size(0)
        H, W = global_features.size()[2:]
        instance_locations = torch.flip(instance_inds, [1])
        instance_locations = instance_locations
        relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1).float()
        relative_coords = (relative_coords / 32).to(dtype=global_features.dtype)
        relative_coords = relative_coords.reshape(n_inst, 2, H, W)
        input_feats = torch.cat((input_feats, relative_coords), dim=1)
        mask = self.conv(input_feats).sigmoid()
        return global_features * mask

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

class Block(nn.Module):
    def __init__(self, inplanes, planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, 1, 1, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out += residual
        out = self.relu(out)

        return out