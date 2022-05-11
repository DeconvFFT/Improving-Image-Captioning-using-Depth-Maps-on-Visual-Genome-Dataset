from helper.pytorch_misc import diagonal_inds
from helper.object_detector import Result

import torch
import torch.nn as nn
import torch.nn.parallel

feature_models = {'v','l','c','d'}#v = visual features, l = location features, c = class features, d = depth features

class BaseRelModel(nn.Module):
    '''
    Base model to predict the predicates usign depth maps using object and subject relationship
    '''
    def __init__(self,total_classes,relation_classes,mode='predcls',number_of_gpus = 1,require_overlap=True,
            active_features=None,frozen_features=None):
        
        super(BaseRelModel,self).__init__()
        self.total_classes = total_classes
        self.relation_classes = relation_classes
        self.number_of_gpus = number_of_gpus
        self.mode = mode
        self.require_overlap = False

        feature_set = self.get_flags(active_features,feature_models,'vcl')

        self.visual_or_not = 'v' in feature_set
        self.loc_or_not = 'l' in feature_set
        self.class_or_not = 'c' in feature_set
        self.depth_or_not = 'd' in feature_set

    #return total number of classes
    @property
    def number_of_classes(self):
        return len(self.total_classes)
    
    #return total relations between subjects and objects
    @property
    def number_of_relations(self):
        return len(self.relation_classes)

    @staticmethod
    def get_flag(input_string):
        flags_set = set(input_string.strip().lower())
        return flags_set

    
    def ground_truth_boxes(self,offset_image,gt_box_list = None,gt_class_list=None,gt_rels_list=None):
        image_inds = gt_class_list[:,0] - offset_image
        region_of_interest = torch.cat((image_inds.float()[:,None],gt_box_list),1)
        labels = gt_class_list[:,1]
        relation_labels = None
        return region_of_interest,labels,relation_labels

    def get_prior_results(self,offset,gt_box_list,gt_class_list,gt_rels_list):
        region_of_interest,object_labels,relation_labels = self.ground_truth_boxes(offset,gt_box_list,gt_class_list,gt_rels_list)
        image_inds = region_of_interest[:,0].long().contiguous() + offset
        box_prior = region_of_interest[:,1:]

        return Result(
            od_box_priors=box_prior,
            rm_box_priors=box_prior,
            od_obj_labels = object_labels,
            rm_obj_labels = object_labels,
            rel_labels = relation_labels,
            im_inds = image_inds
        )

    def get_region_of_interest_and_relations(self,result,offset,gt_box_list,gt_class_list,gt_rels_list):
        image_inds = result.im_inds - offset
        box_list = result.rm_box_priors
        relation_inds = self.get_relation_inds(result.rel_labels,image_inds,box_list)
        region_of_interest = torch.cat((image_inds[:,None].float(),box_list),1)
        return region_of_interest,relation_inds
    
    def get_relation_inds(self,relation_inds,image_inds,box_prior):
        relation_candidates = image_inds.data[:,None] == image_inds.data[None]
        relation_candidates.view(-1)[diagonal_inds(relation_candidates)] = 0

        relation_candidates = relation_candidates.nonzero()
        if relation_candidates.dim() == 0:
            relation_candidates = image_inds.data.new(1,2).fill_(0)
        relation_inds = torch.cat((image_inds.data[relation_candidates[:,0]][:,None],relation_candidates),1)
        return relation_inds

    def __getitem__(self,batch):
        if self.number_of_gpus==1:
            return self(*batch[0])
    
    




