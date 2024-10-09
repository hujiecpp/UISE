import torch
from torch import nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.data import MetadataCatalog
from .neck import UISENeck
from .head import UISEHead
from .loss import VideoSetCriterion, VideoHungarianMatcher
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess


__all__ = ["VideoUISE"]

@META_ARCH_REGISTRY.register()
class VideoUISE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.in_features = cfg.MODEL.UISE.IN_FEATURES
        self.num_classes = cfg.MODEL.UISE.NUM_CLASSES
        self.num_proposals = cfg.MODEL.UISE.NUM_PROPOSALS
        self.object_mask_threshold = cfg.MODEL.UISE.TEST.OBJECT_MASK_THRESHOLD
        self.overlap_threshold = cfg.MODEL.UISE.TEST.OVERLAP_THRESHOLD
        self.metadata =  MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        
        self.backbone = build_backbone(cfg)
        self.size_divisibility = cfg.MODEL.UISE.SIZE_DIVISIBILITY
        if self.size_divisibility < 0:
            self.size_divisibility = self.backbone.size_divisibility
        
        self.sem_seg_postprocess_before_inference = True

        self.num_queries = cfg.MODEL.UISE.NUM_PROPOSALS

        class_weight = cfg.MODEL.UISE.CLASS_WEIGHT
        dice_weight = cfg.MODEL.UISE.DICE_WEIGHT
        mask_weight = cfg.MODEL.UISE.MASK_WEIGHT

        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.UISE.TRAIN_NUM_POINTS,
        )
        
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        loss_list = ["labels", "masks"]

        criterion = VideoSetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.MODEL.UISE.NO_OBJECT_WEIGHT,
            losses=loss_list,
            num_points=cfg.MODEL.UISE.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.UISE.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.UISE.IMPORTANCE_SAMPLE_RATIO,
        )
        self.criterion = criterion        
        self.uise_neck = UISENeck(cfg=cfg, backbone_shape=self.backbone.output_shape()) # 
        self.uise_head = UISEHead(cfg=cfg, num_stages=cfg.MODEL.UISE.NUM_STAGES, criterion=criterion) # 

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.to(self.device)

    def forward(self, batched_inputs):

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        backbone_feats = self.backbone(images.tensor)            
        # print(features)
        features = list()
        for f in self.in_features:
            features.append(backbone_feats[f])
        # outputs = self.sem_seg_head(features)
        neck_feats = self.uise_neck(features)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            # # bipartite matching-based loss
            # losses = self.criterion(outputs, targets)

            losses, cls_scores, mask_preds = self.uise_head(neck_feats, targets)
            return losses
        else:
            losses, cls_scores, mask_preds = self.uise_head(neck_feats, None)
            mask_cls_results = cls_scores #outputs["pred_logits"]
            mask_pred_results = mask_preds #outputs["pred_masks"]
            mask_cls_result = mask_cls_results[0]
            #
            # print(mask_pred_results.shape) # bs*T, 100, h, w
            bs = 1 #
            _, q, h, w = mask_preds.shape #
            mask_pred_results = mask_preds.reshape(bs, -1, q, h, w).permute(0, 2, 1, 3, 4) #
            # upsample masks
            mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                mask_pred_results[0],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            # del outputs
            
            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation
            
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            
            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width)

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

            # print(gt_classes_per_video)

        return gt_instances

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )
 
            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []
 
        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output
        