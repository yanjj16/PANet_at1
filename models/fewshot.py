"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .vgg import Encoder
from .coder import Encoder
from .coder import Decoder

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder()),]))  # 
        self.decoder = Decoder() #
        self.fore_imgs_PA = PALayer(2048)
        # self.back_imgs_PA = PALayer(2048)

# pre_mask 
    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs , pre_mask):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            pre_mask: previous masks of query images
                N X [B X H X W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        obj_nums = len(fore_mask)
        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                +[torch.cat(qry_imgs, dim=0),], dim=0)
        # pre_mask_concat = torch.cat(pre_mask, dim=0)

        all_img_fts = self.encoder(imgs_concat) #
        img_fts = all_img_fts[4]                # 
        qurey_feature = []                      #   
        all_feature_size = []                   # 
        for i in range(5):                      # 
            fts_size = all_img_fts[i].shape[-2:]
            all_feature_size.append(fts_size)
            qurey_feature.append(all_img_fts[i][n_ways * n_shots * batch_size:].view(
                n_queries, batch_size, -1, *fts_size))

        fts_size = img_fts.shape[-2:]
        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'
        ###### Compute loss ######
        align_loss = 0
        outputs = []
        for epi in range(batch_size):
            ###### Extract prototype ######
            att_feature = self.fore_imgs_PA(supp_fts[0, 0, [epi]])  # one way one shot
            supp_fg_fts = [[self.getFeatures(att_feature, fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range( obj_nums)]
            supp_bg_fts = [[self.getFeatures(att_feature,back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(1)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
            # we need to reduce the number of fg_prototypes to one,cause decoder need a fixed channel
            prototypes = [bg_prototype,] + fg_prototypes
            dist_2 = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist_2, dim=1)
            r5 = torch.unsqueeze(qurey_feature[4][0][epi], dim=0)
            r4 = torch.unsqueeze(qurey_feature[3][0][epi], dim=0)
            r3 = torch.unsqueeze(qurey_feature[2][0][epi], dim=0)
            r2 = torch.unsqueeze(qurey_feature[1][0][epi], dim=0)
            pre_mask_concat = torch.unsqueeze(pre_mask[0], dim=1)
            out_tmp = []
            for i in range(obj_nums + 1):
                part_pred = torch.stack([dist_2[i]]*16,dim=1)
                out_tmp.append(self.decoder(r5, part_pred, r4, r3, r2, pre_mask_concat[[epi]]))
            out_tmp = torch.cat(out_tmp,dim=1)
            # print(out_tmp.shape)
            outputs.append(out_tmp)

            # outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / batch_size


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        with torch.no_grad():
            masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
                         / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C

        return masked_fts


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts])
        return fg_prototypes, bg_prototype


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        # n_ways, n_shots = len(fore_mask), len(fore_mask[0])
        n_ways, n_shots = fore_mask.shape[0], fore_mask.shape[1]
        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C
        qry_prototypes = [qry_prototypes[[i]] for i in range(qry_prototypes.shape[0])]
        # Compute the support loss
        loss = 0
        for shot in range(n_shots):
            img_fts = supp_fts[0, [shot]]
            supp_dist = [self.calDist(img_fts, prototype) for prototype in qry_prototypes]
            supp_pred = torch.stack(supp_dist, dim=1)
            supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                      mode='bilinear')
            # Construct the support Ground-Truth segmentation
            supp_label = torch.full_like(fore_mask[0, shot], 255,
                                         device=img_fts.device).long()
            for way in range(n_ways):
                supp_label[fore_mask[way, shot] == 1] = way+1
            supp_label[back_mask[0, shot] == 1] = 0

            # Compute Loss
            loss = loss + F.cross_entropy(
                supp_pred, supp_label[None, ...], ignore_index=255) / n_shots

        return loss
