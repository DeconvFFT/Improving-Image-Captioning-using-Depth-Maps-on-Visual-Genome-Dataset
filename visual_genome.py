import json
import os
from random import shuffle

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from image_transforms import SquarePad, DepthNormalize, RepeatTensor
from helper.blob import Blob
from helper.bbox_helper.bbox import bbox_overlaps
# -- import depth parameters
from helper.config import VG_IMAGES, VG_DEPTH_IMAGES, IM_DATA_FN, VG_SGG_FN, VG_SGG_DICT_FN, BOX_SCALE, IM_SCALE
# from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
#     RandomOrder, Hue, random_crop, DepthNormalize, RepeatTensor
from collections import defaultdict


class VisualGenome(Dataset):
    def __init__(self, roi_file=VG_SGG_FN, sg_file=VG_SGG_DICT_FN,image_file=IM_DATA_FN, 
                filter_duplicate_rels= True, mode='train', n_images = -1, val_imgs=5000,
                filter_non_overlap=True,use_depth=False, three_channels_depth=False):
        '''
        Parameters:
        -----------
        roi_file: HDF5
            HDF5 containing Ground truth bounding boxes, classes and relationships

        sg_file: JSON
            JSON containing scene graph information
        
        image_file: HDF5
            HDF5 
        
        filter_duplicate_rels: bool
            Whether we have duplicate relations or not. If so, we will sample relationships
        
        filter_non_overlap: bool
            Remove non overlapping relations
        
        use_depth: bool
            Add depth maps to images or not

        n_images: int
            Number of training images
        
        val_images: int
            Number of validation images

        Returns:
        --------
        None
        '''

        self.roi_file = roi_file
        self.sg_file = sg_file
        self.image_file = image_file
        self.mode =mode
        self.filter_duplicate_rels=filter_duplicate_rels and self.mode=='train' # only filter duplicates when training


        # load information from scene graph file
        self.split_mask, self.gt_boxes, self.gt_classes,self.relationships = load_graphs(
            self.roi_file, self.mode, n_images, val_imgs
        )

        self.filenames = join_filepaths(image_file, VG_IMAGES, is_depth=False)
        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]

        self.idx_to_classes, self.idx_to_predicates = load_info(sg_file)

        self.rpn_rois = None 

        # define image transforms
        tform = [
            SquarePad(), # square padding
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # using values from the original paper
        ]

        self.use_depth = use_depth
        self.three_channels_depth = three_channels_depth

        if self.use_depth:
            self.depth_filenames = join_filepaths(image_file, VG_DEPTH_IMAGES, is_depth=True)
            self.depth_filenames = [self.depth_filenames[i] for i in np.where(self.split_mask)[0]]
        
            depth_tform = [
                SquarePad(single_channel=True),
                Resize(IM_SCALE),
                ToTensor(),
                DepthNormalize()
            ]

            if three_channels_depth:
                depth_tform.append(RepeatTensor())

            self.depth_transform_pipeline = Compose(depth_tform)
        else:
            self.depth_filenames = None
            self.depth_transform_pipeline = None
        

    @property
    def is_train(self):
        return self.mode.startswith('train')
    
    @classmethod
    def splits(cls, *args, **kwargs):
        '''
        Creates train, validation and test datsets and returns them
        Parameteters:
        ------------
        cls: Class
            Dataset Class
        
        *args: any
            Function arguments
        
        **kwargs: any
            Keyword arguments

        Returns:
        -------
        train: Dataset
            Train Dataset

        val: Dataset
            Validation Dataset

        test: Dataset
            Test Dataset
        '''
        train = cls('train', *args, **kwargs)
        val = cls('val', *args, **kwargs)
        test = cls('test',*args, **kwargs)

        return train, val, test
    

    def __getitem__(self, index):
        '''
        overloads __get_item__ method

        Parameters:
        -----------
        index: int
        Index at which you want to retrieve the item

        Returns:
        ---------

        '''

        image = Image.open(self.filenames[index]).convert('RGB')
        gt_boxes = self.gt_boxes[index].copy()

        w,h = image.size()
        box_scale = BOX_SCALE/max(w,h)

        # setting appropriate scale for images
        img_scale = IM_SCALE/max(w,h)

        if h > w:
            img_shape = (IM_SCALE, int(w * img_scale), img_scale)
        elif h < w:
            img_shape = (int(h * img_scale), IM_SCALE, img_scale)
        else:
            img_shape = (IM_SCALE, IM_SCALE, img_scale)
        
        # ground truth relations
        gt_rels = self.relationships[index].copy()


        # removing duplicate relations
        if self.filter_duplicate_rels:
            rel_set = defaultdict(list)

            for (object1, object2, relation) in gt_rels:
                rel_set[(object1, object2)].append(relation)
            
            gt_rels = [(k[0], k[1], np.random.choice(v)) for k,v in rel_set.items()]
            gt_rels = np.array(gt_rels)
        
        entry = {
            'img': self.transform_pipeline(image),
            'img_size': img_shape,
            'gt_boxes': gt_boxes,
            'gt_classes': self.gt_classes[index].copy(),
            'gt_relations': gt_rels,
            'scale': IM_SCALE / BOX_SCALE,  # Multiply the boxes by this.
            'index': index,
            'flipped': False,
            'fn': self.filenames[index],
        }


        # similar process for depth images
        if self.use_depth:
            depth_image = Image.open(self.depth_filenames[index])
            transformed_depth = self.depth_transform_pipeline(depth_image)
        
            depth_entry = {
                'depth_img': transformed_depth,
                'depth_fn': self.depth_filenames[index]
            }

            entry.update(depth_entry)



        return entry

    def __len__(self):
        return len(self.filenames)
    
    @property
    def num_predicates(self):
        return len(self.idx_to_predicates)

    @property
    def num_classes(self):
        return len(self.idx_to_classes)
    

def join_filepaths(image_file, image_dir=VG_IMAGES, is_depth=False):

    with open(image_file, 'r') as f:
        img_data = json.load(f)

    filenames = []
    extension = 'png' if is_depth else 'jpg'

    for img in img_data:
        filename = '{}.{}'.format(img['image_id'], extension)
        filename = os.path.join(image_dir, filename)

        if os.path.exists(filename):
            filenames.append(filename)
    
    return filenames


def load_graphs(graphs_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
                filter_non_overlap=False):
    roi_h5 = h5py.File(graphs_file, 'r')
    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0
    split_mask = data_split == split

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if mode == 'val':
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert mode == 'train'
            inters = bbox_overlaps(boxes_i, boxes_i)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, relationships

def load_info(info_file):
    '''
    Parameters:
    -----------
    info_file: JSON
        file containing visual genome label meanings
    Returns:
    --------
    ind_to_classes: list
        sorted list of classes
    
    ind_to_predicates: list
        sorted list of predicates
    
    '''
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates

# adding data collate function to enable 
# muilti gpu data sharing https://stackoverflow.com/questions/65932328/pytorch-while-loading-batched-data-using-dataloader-how-to-transfer-the-data-t
def collate(data, n_gpus=1, is_train=False, mode='det', use_depth=False):
    '''
    Parameters:
    -----------
    data: any
        data from train/validation/test dataset

    n_gpus: int
        number of gpus

    is_train: bool
        whether the dataloader is of train, validation or test
    
    mode: str
        mode of collation
    
    use_depth: bool
        Use depthmap or not    

    Returns:
    --------
    blob: Blob

    '''
    blob = Blob(mode=mode,is_train=is_train, num_gpus=n_gpus, batch_size_per_gpu=len(data)//use_depth)
    for d in data:
        blob.append(d)
    blob.reduce()
    return blob

class VGDataloader(torch.utls.data.DataLoader):
    '''
    Iterates through the data and loads it on the GPUs
    '''

    @classmethod
    def splits(cls, train_data, val_data, batch_size=1, n_workers=1, n_gpus=1, mode='det', use_depth=False, **kwargs):
        
        # dataloaders in pytorch: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        # train dataloader
        train_loader = cls(
            dataset=train_data,
            batch_size = batch_size * n_gpus, 
            shuffle=True,
            num_workers = n_workers, 
            collate_fn=lambda x: collate(x, mode=mode, n_gpus=n_gpus, is_train=True, use_depth=use_depth),
            drop_last=True, 
            **kwargs,
        )

        # validation dataloader
        val_loader = cls(
            dataset=val_data,
            batch_size = batch_size * n_gpus, 
            shuffle=True,
            num_workers = n_workers, 
            collate_fn=lambda x: collate(x, mode=mode, n_gpus=n_gpus, is_train=False, use_depth=use_depth),
            drop_last=True, 
            **kwargs,
        )
    
        return train_loader, val_loader