from .cid import CID
from .utils import test_image_process, oks_nms
from .vis import draw_results
import torchvision
from yacs.config import CfgNode
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)

def get_cfg() -> CfgNode:
    from .config import _C

    return _C.clone()

class HumVis(object):
    def __init__(self, backbone_name='hrnet32'):
        cfg = get_cfg()
        cfg.defrost()
        cfg.MODEL.BACKBONE = backbone_name
        cfg.freeze()
        model = CID(cfg, device)
        model.load_state_dict(torch.load('model_files/humvis.pth', map_location='cpu'), strict=True)
        model = model.to(device)
        self.model = model.eval()

        self.init_video_queue()
    
    def init_video_queue(self):
        self.instances = []

    @torch.no_grad()
    def get_analysis_results(self, img, tracking=False):
        assert len(img.shape) == 3 and img.shape[2] == 3
        # convert bgr to rgb
        img_rgb = img[:, :, ::-1] - np.zeros_like(img)
        ori_size = img.shape[:2]
        img_rgb, padded_size, resized_size, scale_factor = test_image_process(img_rgb)
        transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        inputs = transforms(img_rgb).unsqueeze(0)
        # get pose, mask, part results
        inputs = inputs.to(device)
        results = self.model(inputs, input_scale=1/scale_factor)
        if 'centers' not in results:
            return img, img, img, img

        poses = results['poses'].cpu().numpy()
        poses[:, :, :2] /= scale_factor

        # nms by pose
        scores = results['pose_scores'].cpu().numpy()
        keep, _ = oks_nms(poses, scores,  0.9, np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0)

        poses = poses[keep]

        masks = results['masks'][keep]
        # upsample mask
        masks = F.interpolate(masks,
                                size=padded_size,
                                mode='bilinear')[:, :, :resized_size[0], :resized_size[1]]
        masks = F.interpolate(masks,
                                size=ori_size,
                                mode='bilinear')
        
        masks = masks[:, 0, :, :]
        masks = (masks > 0.5).float()
        masks = masks.cpu().numpy()

        partmaps = results['parts'][keep]
        partmaps = F.interpolate(partmaps,
                                  size=padded_size,
                                  mode='bilinear')[:, :, :resized_size[0], :resized_size[1]]
        partmaps = F.interpolate(partmaps,
                                size=ori_size,
                                mode='bilinear')
        partmaps = partmaps.max(dim=1)[1]
        partmaps = partmaps.cpu().numpy()

        meshes = results['mesh'][keep]
        mesh_locs = results['mesh_loc'][keep]

        if poses.shape[0] > 1 and tracking:
            instance_feats = results['feats'][keep]
            
            # first frame
            if len(self.instances) == 0:
                for i in range(instance_feats.shape[0]):
                    self.instances.append(instance_feats[i].cpu().numpy())
                target_ids = list(range(len(self.instances)))
            else:
                # perform ID match
                src_feats = np.stack(self.instances, axis=0)
                assert src_feats.shape[0] == len(self.instances)
                tgt_feats = instance_feats.cpu().numpy()
                match_mat = np.matmul(src_feats, tgt_feats.T)
                src_ind, tgt_ind = linear_sum_assignment(1 - match_mat)
                target_ids = [-1 for _ in range(tgt_feats.shape[0])]
                for i in range(len(tgt_ind)):
                    target_ids[tgt_ind[i]] = src_ind[i]
                    # update feature
                    self.instances[src_ind[i]] = tgt_feats[tgt_ind[i]]
                # update new feature
                if len(self.instances) < 50:
                    for i in range(len(target_ids)):
                        if target_ids[i] == -1:
                            target_ids[i] = len(self.instances)
                            self.instances.append(tgt_feats[i])
            instance_ids = target_ids
        else:
            instance_ids = list(range(poses.shape[0]))

        pose_img, mask_img, part_img, mesh_img = draw_results(img, poses, masks, partmaps, meshes, mesh_locs, self.model.iia.smpl.faces_tensor, instance_ids=instance_ids)

        return pose_img, mask_img, part_img, mesh_img

