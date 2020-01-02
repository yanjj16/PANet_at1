import numpy as np
from PIL import Image
import torch
import os

import torch.utils.data as data
from glob import glob
from .common import BaseDataset

class davis2017(BaseDataset):
    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super(davis2017, self).__init__(base_dir)
        self.split = split
        self.annFile = f'{base_dir}/annotations.txt'  # train.txt
        self.transforms = transforms
        self.to_tensor = to_tensor
        self.base_dir = base_dir
    def __len__(self):     
        count = 0
        for _,_ in enumerate(open(self.annFile,'rU')):
            count = count + 1
        return  count

    def catId(self, item):
        """
        :param class_name: 
        :return:  id_cla 
        """
        cat_dir = self.base_dir + '/Annotations/480p'
        class_list = os.listdir(cat_dir)
        class_id = item.split('/')[0]
        id_dic = class_list.index(class_id) + 1
        return  id_dic

    def load_anno(self,item):
        pass

    def get_Imgids(self,class_id):   
        base_dir = self.base_dir + '/Annotations/480p'
        class_list = os.listdir(base_dir)
        class_name = class_list[class_id - 1]
        image_dir = self.base_dir + f'/JPEGImages/480p/{class_name}'
        img_ids = []
        for _, _, img_id in os.walk(image_dir):
            for way in img_id:
                way = class_name + '/' + way
                way = way.strip('\n')
                img_ids.append(way)
        return sorted(img_ids)   # img_ids is the name list of a certain class

    def P2msks(self, Img, obj_ids):
        img = np.array(Img)
        Imgs = []
        for idx in obj_ids:
            label = Image.fromarray((img == idx) * 255.0).convert('L')
            Imgs.append(np.array(label))
        return Imgs,img.shape

    def msks2P(self, msks, obj_ids, img_size):
        # if max_num == 1:
        #     return msks[0]
        if len(msks) != len(obj_ids):
            print('error, len(msks) != len(objs_ids)')
        if obj_ids[0] != -1:
            P = np.zeros(msks[0].shape)
        elif obj_ids[0] == -1:
            P = np.zeros(img_size)
        else:
            print("error")
        for idx, msk in enumerate(msks):
            ids = np.nonzero(msk)
            if len(obj_ids) > 0:
                for i in range(len(ids[0])):
                    P[ids[0][i], ids[1][i]] = idx + 1
                   # distinguish different objects by idx+1
        return P

    def __getitem__(self, item):  
        # id = self.catId()
        # anno_list = []
        # with open(self.annFile) as f:
        #     for line in f.readlines():
        #         line = line.strip('\n')
        #         anno_list.append(line)
        # Open Image
        image = Image.open(f'{self.base_dir}/JPEGImages/480p/{item}')   
        if image.mode == 'L':
            image = image.convert('RGB')

        # Process masks  include current mask and pre mask
        # semantic_masks = {}
        mask_item = item.split('.')[0]+'.png'

        cla_id = self.catId(item)
        semantic_mask = Image.open(f'{self.base_dir}/Annotations/480p/{mask_item}')

        obj_ids = list(set(np.asarray(semantic_mask).reshape(-1)))    # new ;number of objects in a picture
        obj_ids.sort()
        obj_ids = obj_ids[1:]    # to filter the background mask
        if len(obj_ids) == 0:
            obj_ids = [-1,]
        semantic_mask,img_size = self.P2msks(semantic_mask,obj_ids)
        semantic_mask = self.msks2P(semantic_mask,obj_ids,img_size)
        semantic_mask = Image.fromarray(semantic_mask)
        semantic_masks = {cla_id:semantic_mask}
        pre_base_dir = item.split('/')[1]
        pre_base_dir = pre_base_dir.split('.')[0]
        pre_num = int(pre_base_dir)
        if pre_num ==0:
            pre_num = pre_num
        else:
            pre_num = pre_num - 1
        pre_num = str(pre_num)
        pre_num = pre_num.rjust(5,'0') + '.png'
        pre_num = item.split('/')[0] + '/' +pre_num
        pre_base_dir = f'{self.base_dir}/Annotations/480p/{pre_num}'
        pre_semantic_mask = Image.open(pre_base_dir)
        pre_semantic_mask = self.P2msks(pre_semantic_mask, obj_ids)
        pre_semantic_mask = self.msks2P(pre_semantic_mask, obj_ids)
        pre_semantic_mask = Image.fromarray(pre_semantic_mask)
        pre_semantic_masks = {cla_id: pre_semantic_mask}
        for i in range(len(obj_ids)):
            obj_ids[i] = int(obj_ids[i])
        sample = {'image':image,
                  'pre_label':pre_semantic_masks,
                  'label':semantic_masks,
                  'obj_ids':obj_ids,    # list of objects id
                  'img_size':img_size,
                  'label_t':semantic_masks}
        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without mean subtraction/normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))

        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = item       
        sample['image_t'] = image_t
         
        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix]) 
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample

# to generate  Davis annotations
def generate_ann_list(self, base_dir=None):
    base_dir = 'D:/Dataset/DAVIS/Annotations/480p'
    class_list = os.listdir(base_dir)
    a = 'DAVIS/Annotations/480p'
    fw = open('D:/Dataset/DAVIS/annotations.txt',mode = 'w')
    # b = 0
    for clist in class_list:
        dir = base_dir + '/' + clist
        # a is root;b is folder name; c is file name
        for _, _, c in os.walk(dir):
            for name in c:
                ann_name = a + '/' + clist + '/' + name
                ann_name = clist + ';' + ann_name
                #b = b + 1
                fw.write(ann_name +'\n')
    fw.close()
    return


if __name__ == '__main__':
    base_dir = 'D:/Dataset/DAVIS/Annotations/480p'
    generate_ann_list(base_dir)