import numpy as np
from torch import optim
import torch
import time
import pandas as pd
import os
from visual_genome import VGDataLoader, VG 

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from helper.pytorch_misc import remove_params,optimistic_restore, clip_grad_norm
from helper.config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional  as F
from helper.sg_eval import BasicSceneGraphEvaluator

config = ModelConfig()

if config.rnd_seed!=None:
    np.random.seed(config.rnd_seed)
    torch.manual_seed(config.rnd_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.rnd_seed)
        torch.cuda.manual_seed(config.rnd_seed)

if config.config == 'shz_fusion':
    from rel_model_fusion import RelModel
else:
    raise ValueError("Provide valid fusion model")

writer = SummaryWriter(comment='_run#'+ config.save_dir.split('/')[-1])

train, val, _ = VG.splits(num_val_im=config.val_size, filter_duplicate_rels=True,
                          use_proposals=config.use_proposals,
                          filter_non_overlap=config.mode == 'sgdet',
                          # -- Depth dataset parameters
                          use_depth=config.load_depth,
                          three_channels_depth=config.pretrained_depth)

TrainLoader, ValidationLoader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               num_gpus=config.num_gpus,
                                               # -- Depth dataset parameters
                                               use_depth=config.load_depth)

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=config.num_gpus, mode=config.mode, require_overlap_det=True,
                    use_resnet=config.use_resnet, order=config.order,
                    nl_edge=config.nl_edge, nl_obj=config.nl_obj, hidden_dim=config.hidden_dim,
                    use_proposals=config.use_proposals,
                    pass_in_obj_feats_to_decoder=config.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=config.pass_in_obj_feats_to_edge,
                    pooling_dim=config.pooling_dim,
                    rec_dropout=config.rec_dropout,
                    use_bias=config.use_bias,
                    use_tanh=config.use_tanh,
                    use_vision=config.use_vision,
                    # -- The proposed config parameters
                    depth_config=config.depth_config,
                    pretrained_depth=config.pretrained_depth,
                    active_features=config.active_features,
                    frozen_features=config.frozen_features,
                    use_embed=config.use_embed)

#Freezing the detector

for n, param in detector.detector.named_parameters():
    param.requires_grad = False


def is_conv_param_depth(name):
    if config.depth_config == 'resnet18':
        depth_conv_params = ['depth_backbone','depth_rel_head','depth_rel_head_union','depth_union_boxes']
    else:
        raise ValueError('Please provide "resnet18" model')
    
    for param in depth_conv_params:
        if name.startswith(param):
            return True
    return False

def get_optimizer(learning_rate):
    fc_parameters = [p for n,p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_parameters = [p for n,p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]

    params = [{'params':fc_parameters,'lr':learning_rate/10.0},{'params':non_fc_parameters}]
    optimizer = optim.Adam(params,lr = learning_rate)

    scheduler = ReduceLROnPlateau(optimizer,'max',patience=6,factor=0.1,verbose=True, threshold=0.0001,threshold_mode='abs',cooldown=1)
    return optimizer, scheduler

remove_parameters = ['rel_out.bias','rel_out.weight','fusion_hlayer.bias','fusion_hlayer.weight']

start_epoch = -1
if config.ckpt is not None:
    ckpt = torch.load(config.ckpt)
    start_epoch = ckpt['epoch']
    if not config.keep_weights:
        remove_params(ckpt['state_dict'],remove_parameters)
    if not optimistic_restore(detector,ckpt['state_dict']):
        start_epoch = -1
detector.cuda()

def training_epoch(epoch_num):
    detector.train()
    train_list = []
    start=time.time()
    
    for b, batch in enumerate(TrainLoader):
        train_list.append(training_batch(batch,verbose=b%(config.print_interval*10)==0))
    if b % config.print_interval == 0 and b >= config.print_interval:
            mn = pd.concat(train_list[-config.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / config.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(TrainLoader), time_per_batch, len(TrainLoader) * time_per_batch / 60))
            print(mn)

            print('-----------', flush=True)
            start = time.time()


def training_batch(b):
    '''
    b contains all the information of the image batch, bounding boxes, etc
    '''
    result = detector[b]
    loss_history = {}
    loss_history['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_lables)
    loss_history['val_loss'] = F.cross_entropy(result.rel_dists, result.rel_labels[:,-1])
    loss = sum(loss_history.values())

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n,p) for n,p in detector.named_parameters() if p.grad is not None],max_norm=config.clip,verbose=False,clip=True)

    loss_history['total'] = loss
    optimizer.step()
    res = pd.Series({x:y.items() for x,y in loss_history.items()})
    return res

def validation_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_nodes()
    for val_b,batch in enumerate(ValidationLoader):
        validation_batch(config.num_gpus * val_b,batch,evaluator)
    evaluator[config.mode].print_stats(epoch,writer)
    return np.mean(evaluator[config.mode].rsult_dict[config.mode + '_recall'][100])

def validation_batch(batch_num,b,evaluator):
    det_res = detector[b]
    if config.num_gpus == 1:
        det_res = [det_res]
    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        evaluator[config.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
print('Training starts!')

optimizer,scheduler = get_optimizer(config.lr *  config.num_gpus * config.batch_size)

for epoch in range(start_epoch + 1,start_epoch+1 + config.num_epochs):
    rez = training_epoch(epoch)

    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)

    if config.save_dir!=None:
        torch.save({
            'epoch':epoch,
            'state_dict':detector.state_dict(),
        },os.path.join(config.save_dir,'{}-{}.tar'.format('vgrel',epoch)))
    val_MAP = validation_epoch()
    scheduler.step(val_MAP)

                    








