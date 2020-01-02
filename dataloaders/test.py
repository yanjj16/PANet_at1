import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from .common import PairedDataset
from .davis2017 import davis2017

def attrib_basic(_sample, class_id):
    """
    Add basic attribute
    Args:
        _sample: data sample
        class_id: class label asscociated with the data
            (sometimes indicting from which subset the data are drawn)
    """
    return {'class_id': class_id}

def get_fg_mask(label, class_id):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask

    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    return {'fg_mask': fg_mask}

def get_bg_mask(label,class_ids):
    bg_mask = torch.ones_like(label)
    for class_id in class_ids:
        bg_mask[label == class_id] = 0
    return  {'bg_mask': bg_mask}

def fewShot(paired_sample, n_ways, n_shots, cnt_query, davis=True):
    """
    Postprocess paired sample for fewshot settings

    Args:
        paired_sample:
            data sample from a PairedDataset
        n_ways:
            n-way few-shot learning
        n_shots:
            n-shot few-shot learning
        cnt_query:
            number of query images for each class in the support set
    """
    ###### Compose the support and query image list ######
    cumsum_idx = np.cumsum([0,] + [n_shots + x for x in cnt_query])


    class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)]

    # support images
    support_images = [[paired_sample[cumsum_idx[i] + j]['image'] for j in range(n_shots)]
                      for i in range(n_ways)]
    support_num_objs = [[paired_sample[cumsum_idx[i] + j]['obj_ids'] for j in range(n_shots)]
                      for i in range(n_ways)]
    support_num_objs = support_num_objs[0][0]  #
    support_images_t = [[paired_sample[cumsum_idx[i] + j]['image_t'] for j in range(n_shots)]
                        for i in range(n_ways)]

    # support image labels
    if davis:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'][class_ids[i]]
                           for j in range(n_shots)] for i in range(n_ways)]
    else:
        raise ValueError("When 'davis=true', you should use davis2017 dataset")

    # query images, masks and class indices
    query_images = [paired_sample[cumsum_idx[i+1] - j - 1]['image'] for i in range(n_ways)
                    for j in range(cnt_query[i])]
    query_images_t = [paired_sample[cumsum_idx[i+1] - j - 1]['image_t'] for i in range(n_ways)
                      for j in range(cnt_query[i])]

    if davis:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['pre_label'][class_ids[i]]
                        for i in range(n_ways) for j in range(cnt_query[i])]
        label_t = [paired_sample[cumsum_idx[i + 1] - j - 1]['label_t'][class_ids[i]]
                        for i in range(n_ways) for j in range(cnt_query[i])]
    else:
        raise ValueError("When 'davis=true', you should use davis2017 dataset")

    ###### Generate support image masks ######
    # need to ensure the following line is right
    support_fg_mask = [[get_fg_mask(support_labels[0][shot], support_num_objs[way])
                     for shot in range(n_shots)] for way in range(len(support_num_objs))]
    support_bg_mask = [[get_bg_mask(support_labels[way][shot],support_num_objs)
                        for shot in range(n_shots)] for way in range(n_ways)]
    # ###### Generate query label (class indices in one episode, i.e. the ground truth)######

    query_labels_tmp = [torch.zeros_like(x) for x in label_t]
    for i, query_label_tmp in enumerate(query_labels_tmp):
        query_label_tmp[label_t[i] == 255] = 255
        for j in range(len(support_num_objs)):
            query_label_tmp[label_t[i] == support_num_objs[j]] = j+1  # it doesn't have to be j + 1


    return {'class_ids': class_ids,

            'support_images_t': support_images_t,
            'support_images': support_images,
            'support_fg_mask': support_fg_mask,    # length of support mask is not way
            'support_bg_mask': support_bg_mask,
            # 'support_num_objs':support_num_objs,
            'query_images_t': query_images_t,
            'query_images': query_images,
            'query_labels': query_labels_tmp,
            'query_masks': query_labels,
           }

class davis2017_test_loader(Dataset):
    def __init__(self, datasets, num_objects,base_dir):
        super().__init__()
        self.datasets = datasets
        self.n_datasets = len(self.datasets)
        self.n_data = [len(dataset) for dataset in self.datasets]
        self.num_objects = num_objects
        self.base_dir = base_dir
        val_dir = base_dir + '/ImageSets/2017/val.txt'
        with open(self.base_dir) as f:
            class_name = f.readlines()
        class_names = [i.rstrip('\n') for i in class_name]
        self.class_names = sorted(class_names)
    def __len__(self):
        return self.n_datasets

    def __getitem__(self, idx):
        return self.datasets[idx],self.num_objects[idx],self.class_names[idx]

def davis2017_test(base_dir, split, transforms, to_tensor, labels):
    # load image ids for each class
    davisseg = davis2017(base_dir,split,transforms,to_tensor)
    davisseg.add_attrib('basic',attrib_basic, {})
    sub_ids = [davisseg.get_Imgids(i) for i in labels]
    num_objects = [len(i) for i in sub_ids]
    subsets = davisseg.subsets(sub_ids,[{'basic': {'class_id': i}} for i in labels])
    paired_data = davis2017_test_loader(subsets, num_objects,base_dir)
    return paired_data


if __name__ == '__main__':
    pass
