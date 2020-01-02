import os
import random
import torch
import numpy as np

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
    # bg_mask = torch.where(label != class_id,
    #                       torch.ones_like(label), torch.zeros_like(label))
    # for class_id in class_ids:
    #     bg_mask[label == class_id] = 0

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
        coco:
            MS COCO dataset
    """
    ###### Compose the support and query image list ######
    cumsum_idx = np.cumsum([0,] + [n_shots + x for x in cnt_query]) 


    class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)]

    # support images
    support_images = [[paired_sample[cumsum_idx[i] + j]['image'] for j in range(n_shots)]
                      for i in range(n_ways)]
    support_num_objs = [[paired_sample[cumsum_idx[i] + j]['obj_ids'] for j in range(n_shots)]
                      for i in range(n_ways)]
    support_num_objs = support_num_objs[0][0]  # what's the type?
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

def davis2017_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
                n_queries=1):
    """
    Args:
        base_dir:
            COCO dataset directory
        split:
            which split to use
            choose from ('train', 'val')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            labels of the data   labels is the chosen training classes
        n_ways:
            n-way few-shot learning, should be no more than # of labels
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of pairs
        n_queries:
            number of query images
    """
    # load image ids for each class
    davisseg = davis2017(base_dir,split,transforms,to_tensor)
    davisseg.add_attrib('basic',attrib_basic, {})
    sub_ids = [davisseg.get_Imgids(i) for i in labels]
    subsets = davisseg.subsets(sub_ids,[{'basic': {'class_id': i}} for i in labels])

    # choose the classes of queries
    # choice different n_ways
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries),
                            minlength=n_ways)
    # Set the number of images for each class
    n_elements = 0

    if len(cnt_query>1):
        n_elements = [n_shots + x for x in cnt_query]
    else:
        n_elements = n_shots + cnt_query[0]

    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, test=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query, 'davis': True})])
    return paired_data


if __name__ == '__main__':
    pass