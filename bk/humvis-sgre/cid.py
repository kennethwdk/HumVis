import torch
from torch import nn
import torch.nn.functional as F

from .backbone import build_backbone
from .cid_module import build_iia_module, build_gfd_module

class CID(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.backbone = build_backbone(cfg)
        self.iia = build_iia_module(cfg)
        self.gfd = build_gfd_module(cfg)

        # inference
        self.max_instances = cfg.DATASET.MAX_INSTANCES
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.flip_test = cfg.TEST.FLIP_TEST
        self.flip_index = cfg.DATASET.FLIP_INDEX
        self.part_flip_indx = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
        self.max_proposals = cfg.TEST.MAX_PROPOSALS
        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2
    
    def forward(self, images, input_scale=1.0):
        images = images.to(self.device)

        feats = self.backbone(images)

        results = {}
        if self.flip_test:
            feats[1, :, :, :] = feats[1, :, :, :].flip([2])

        instances = self.iia(feats)
        if len(instances) == 0: return results

        instance_heatmaps, instance_masks, instance_parts, instance_mesh = self.gfd(feats, instances, input_scale=input_scale)

        if self.flip_test:
            # flip heatmap for pose
            instance_heatmaps, instance_heatmaps_flip = torch.chunk(instance_heatmaps, 2, dim=0)
            instance_heatmaps_flip = instance_heatmaps_flip[:, self.flip_index, :, :]
            instance_heatmaps = (instance_heatmaps + instance_heatmaps_flip) / 2.0

            # flip mask
            instance_masks, instance_masks_flip = torch.chunk(instance_masks, 2, dim=0)
            instance_masks = (instance_masks + instance_masks_flip) / 2.0

            # flip part
            instance_parts, instance_parts_flip = torch.chunk(instance_parts, 2, dim=0)
            instance_parts_flip = instance_parts_flip[:, self.part_flip_indx, :, :]
            instance_parts = (instance_parts + instance_parts_flip) / 2.0

        instance_scores = instances['instance_score']
        # get pose
        num_people, num_keypoints, h, w = instance_heatmaps.size()
        center_pool = F.avg_pool2d(instance_heatmaps, 3, 1, 1)
        instance_heatmaps = (instance_heatmaps + center_pool) / 2.0
        nms_instance_heatmaps = instance_heatmaps.view(num_people, num_keypoints, -1)
        vals, inds = torch.max(nms_instance_heatmaps, dim=2)
        x, y = inds % w, (inds / w).long()
        # shift coords by 0.25
        x, y = x.float(), y.float()
        vals = vals * instance_scores.unsqueeze(1)
        poses = torch.stack((x, y, vals), dim=2)
        poses[:, :, :2] = poses[:, :, :2] * 4 + 2
        pose_scores = torch.mean(poses[:, :, 2], dim=1)

        # mask
        masks = instance_masks
        mask_scores = instance_scores
        # part
        parts = instance_parts
        part_scores = instance_scores
        # mesh
        mesh = instance_mesh
        mesh_loc = torch.flip(instances['instance_coord'], [1]) * 4 * input_scale

        results.update({'centers': instances['instance_coord']})
        results.update({'feats': instances['instance_param']})
        results.update({'poses': poses})
        results.update({'pose_scores': pose_scores})
        results.update({'masks': masks})
        results.update({'mask_scores': mask_scores})
        results.update({'parts': parts})
        results.update({'part_scores': part_scores})
        results.update({'mesh': mesh})
        results.update({'mesh_loc': mesh_loc})

        return results