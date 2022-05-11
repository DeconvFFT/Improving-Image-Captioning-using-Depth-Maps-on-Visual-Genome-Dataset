from helper.depth_cnn import DepthCNN, DEPTH_DIMS
from helper.box_utils import center_size
from rel_model_base import BaseRelModel
from helper.pytorch_misc import xavier_init,ScaleLayer
from helper.surgery import filter_dets
from helper.object_detector import ObjectDetector,load_vgg

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.ops import RoIAlign


class RelModel(BaseRelModel):
    FC_SIZE_VISUAL = 512
    FC_SIZE_CLASS = 64
    FC_SIZE_LOC = 20
    FC_SIZE_DEPTH = 4096
    LOC_INPUT_SIZE = 8

    def __init__(self, total_classes, rel_classes, mode='sgdet', number_of_gpus=1, require_overlap=True,
                 embedded_dimension=200, hidden_dimension=4096, use_resnet=False, threshold=0.01,
                 use_proposals=False, depth_model=None, pretrained_depth=False,
                 active_features=None, frozen_features=None, use_embed=False, **kwargs):
        BaseRelModel.__init__(self,total_classes,rel_classes,mode,number_of_gpus,require_overlap,active_features,frozen_features)
        self.pool_size = 7
        self.embedded_dimension = embedded_dimension
        self.hidden_dimension = hidden_dimension
        self.object_dimension = 4096

        self.depth_model = depth_model
        self.pretrained_depth = pretrained_depth
        self.depth_pooling_dim = DEPTH_DIMS[self.depth_model]
        self.use_embed = use_embed
        self.detector = nn.Module()
        feature_size = 0

        if self.visual_or_not and self.loc_or_not and self.class_or_not and self.depth_or_not:
            self.detector = ObjectDetector(
                classes = total_classes,
                mode = 'gtbox',
                use_resnet=use_resnet,
                thresh = threshold,
                max_per_img=64,
            )
            self.roi_fmap_object = load_vgg(pretrained=False).classifier

            self.visual_hiddenlayer = nn.Sequential(*[xavier_init(nn.Linear(self.object_dimension * 2,self.FC_SIZE_VISUAL)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.8)
            ])
            self.visual_scale = ScaleLayer(1.0)
            feature_size += self.FC_SIZE_VISUAL

            self.location_hiddenlayer = nn.Sequential(*[xavier_init(nn.Linear(self.object_dimension*2,self.FC_SIZE_LOC)),
            nn.Relu(inplace=True),
            nn.Dropout(0.1)
            ])
            self.location_scale = ScaleLayer(1.0)
            feature_size += self.FC_SIZE_LOC

            classme_input_dim = self.embedded_dimension if self.use_embed else self.number_of_classes
            self.classme_hiddenlayer = nn.Sequential(*[xavier_init(nn.Linear(classme_input_dim * 2, self.FC_SIZE_CLASS)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1) 
            ])
            self.classme_scale =  ScaleLayer(1.0)
            feature_size+=self.FC_SIZE_CLASS

            self.depth_backbone = DepthCNN(depth_model = self.depth_model,pretrained = self.pretrained_depth)

            self.depth_relation_head = self.depth_backbone.get_classifier()

            self.classme_hiddenlayer = nn.Sequential(*[xavier_init(nn.Linear(self.depth_pooling_dim*2,self.FC_SIZE_DEPTH)),

                nn.Relu(inplace=True),
                nn.Dropout(0.6),
            ])

            self.depth_scale = ScaleLayer(1.0)
            feature_size+=self.FC_SIZE_DEPTH
        else:
            raise ValueError('Please provide "lcvd" with -active_features')

        self.classme_hiddenlayer = nn.Sequential(*[xavier_init(nn.Linear(feature_size,self.hidden_dimension)),
            nn.Relu(inplace=True),
            nn.Dropout(0.1),
        ])

        self.relation_out = xavier_init(nn.Linear(self.hidden_dimension,self.num_rels,bias=True))

    def get_region_of_interest_features(self,features,region_of_interest):
        feature_pool = RoIAlign((self.pool_size,self.pool_size),spatial_sacle=1/16,sampling_ratio=-1)(features,region_of_interest)
        return self.depth_relation_head(feature_pool)

    def get_region_of_interest_features_depth(self, features, region_of_interest):
        feature_pool = RoIAlign((self.pool_size, self.pool_size), spatial_scale=1 / 16, sampling_ratio=-1)(
            features, region_of_interest)
        return self.depth_relation_head(feature_pool)

    @staticmethod
    def get_location_features(boxes,subject_inds,object_inds):
        boxes_centered = center_size(boxes.data)
        center_subject = boxes_centered[subject_inds][:,0:2]
        size_subject = boxes_centered[subject_inds][:,2:4]

        center_object = boxes_centered[object_inds][0,0:2]
        size_object = boxes_centered[object_inds][0,2:4]

        subject_coord = (center_subject - center_object)/size_object
        subject_size = torch.log(size_subject/size_object)

        object_coord = (center_object - center_subject)/center_subject
        object_size  = torch.log(size_object/size_subject)

        location_feature = Variable(torch.cat((
            subject_coord,subject_size,object_coord,object_size
        ),1))

        return location_feature

    def forward(self,x,image_sizes,offset,gt_box_list=None,gt_class_list=None,get_rels_list = None,proposals=None,
            train_anchor_inds=None,return_fmap=False,depth_images=None):

        
        result = self.detector(x,image_sizes,offset,gt_box_list,gt_class_list,get_rels_list,proposals,train_anchor_inds,return_fmap=True)
    

        region_of_interest,relation_inds = self.get_region_of_interest_and_relations(offset,gt_box_list,gt_box_list,gt_class_list,get_rels_list)
        boxes = result.rm_box_priors

        subject_inds = relation_inds[:,1]
        object_inds = relation_inds[:,2]

        result.obj_preds = result.rm_obj_labels

        result.rm_obj_dists = F.one_hot(result.rm_obj_labels.data,self.num_classes).float()
        obj_cls = result.rm_obj_dists
        result.rm_obj_dists = result.rm_obj_dists * 1000 + (1 - result.rm_obj_dists) * (-1000)

        rel_features = []
        
        result.obj_fmap = self.get_region_of_interest_features(result.fmap.detach(),region_of_interest)
        rel_visual = torch.cat((result.obj_fmap[subject_inds],result.obj_fmap[object_inds]),1)
        rel_visual_fc = self.visual_hiddenlayer(rel_visual)
        rel_visual_scale = self.visual_scale(rel_visual_fc)
        rel_features.append(rel_visual_scale)

        rel_location = self.get_location_features(boxes, subject_inds, object_inds)
        rel_location_fc = self.location_hiddenlayer(rel_location)
        rel_location_scale = self.location_scale(rel_location_fc)
        rel_features.append(rel_location_scale)

        rel_classme = torch.cat((obj_cls[subject_inds], obj_cls[object_inds]), 1)
        rel_classme_fc = self.classme_hiddenlayer(rel_classme)
        rel_classme_scale = self.classme_scale(rel_classme_fc)
        rel_features.append(rel_classme_scale)

        depth_features = self.depth_backbone(depth_images)
        depth_rois_features = self.get_region_of_interest_features_depth(depth_features, region_of_interest)

        # -- Create a pairwise relation vector out of location features
        rel_depth = torch.cat((depth_rois_features[subject_inds], depth_rois_features[object_inds]), 1)
        rel_depth_fc = self.classme_hiddenlayer(rel_depth)
        rel_depth_scale = self.depth_scale(rel_depth_fc)
        rel_features.append(rel_depth_scale)
        
        rel_fusion = torch.cat(rel_features,1)
        rel_embeddings = self.classme_hiddenlayer(rel_fusion)
        result.rel_dists = self.relation_out(rel_embeddings)
        if self.training:
            return result
        bboxes = result.rm_box_priors
        rel_rep = F.softmax(result.rel_dists,dim=1)
        return filter_dets(bboxes,result.obj_scores,result.obj_preds,relation_inds[:,1:],rel_rep)


