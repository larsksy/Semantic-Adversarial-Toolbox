from sat.attacks.attack import ColorAttack
import torch
import cv2
from tqdm import tqdm
import numpy as np


class ColorFool(ColorAttack):
    pass


def visualize_result(data, pred, pred_prob, args):
    (img, info) = data
    img_name = info.split('/')[-1]

    ### water mask: water, sea, swimming pool, waterfalls, lake and river
    water_mask = (pred == 21)
    sea_mask = (pred == 26)
    river_mask = (pred == 60)
    pool_mask = (pred == 109)
    fall_mask = (pred == 113)
    lake_mask = (pred == 128)
    water_mask = (water_mask | sea_mask | river_mask | pool_mask | fall_mask | lake_mask).astype(int)
    if args.mask_type == 'smooth':
        water_mask = water_mask.astype(float) * pred_prob

    water_mask = water_mask * 255.
    cv2.imwrite('{}/water/{}.png'.format(args.result, img_name.split('.')[0]), water_mask)

    ### Sky mask
    sky_mask = (pred == 2).astype(int)
    if args.mask_type == 'smooth':
        sky_mask = sky_mask.astype(float) * pred_prob
    sky_mask = sky_mask * 255.
    cv2.imwrite('{}/sky/{}.png'.format(args.result, img_name.split('.')[0]), sky_mask)

    ### Grass mask
    grass_mask = (pred == 9).astype(int)
    if args.mask_type == 'smooth':
        grass_mask = grass_mask.astype(float) * pred_prob

    grass_mask = grass_mask * 255.
    cv2.imwrite('{}/grass/{}.png'.format(args.result, img_name.split('.')[0]), grass_mask)

    ### Person mask
    person_mask = (pred == 12).astype(int)
    if args.mask_type == 'smooth':
        person_mask = person_mask.astype(float) * pred_prob
    person_mask = person_mask * 255.
    cv2.imwrite('{}/person/{}.png'.format(args.result, img_name.split('.')[0]), person_mask)


def test(segmentation_module, loader, args):
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
#        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores += (pred_tmp.cpu() / len(args.imgSize))

            pred_prob, pred = torch.max(scores, dim=1)
            pred = np.array(pred.squeeze(0).cpu())
            pred_prob = np.array(pred_prob.squeeze(0).cpu())

        # visualization
        visualize_result((batch_data['img_ori'], batch_data['info']), pred, pred_prob, args)

        pbar.update(1)
