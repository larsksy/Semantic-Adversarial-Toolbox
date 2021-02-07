from sat.data.datasets import DatasetBuilder, TestDataset
from sat.models.ModelBuilder import ResnetDilated, PPMDeepsup
from sat.models.Segmentation import SegmentationModule
from sat.attacks.color.colorfool import test
from torch.utils.data import DataLoader
from torch.utils import collect_env
import torch.nn as nn
import torch
import fnmatch
import os
import argparse


def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, nargs='+', type=str,
                        help='a list of image paths, or a directory name')
    #parser.add_argument('--model_path', required=True,
    #                    help='folder to model path')
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet50dilated',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    # Misc arguments
    parser.add_argument('--result', default='.',
                        help='folder to output visualization results')
    parser.add_argument('--mask_type', required=True,
                        help='Type 0f mask: binary or smooth')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id for evaluation')
    args = parser.parse_args()

    print(collect_env.get_env_info())

    print(torch.cuda.is_available())

    # Dataset and Loader
    if len(args.dataset) == 1 and os.path.isdir(args.dataset[0]):
        test_imgs = find_recursive(args.dataset[0], ext='.*')
    else:
        test_imgs = args.dataset
    list_test = [{'fpath_img': x} for x in test_imgs]
    print(list_test)
    dataset_test = TestDataset(list_test, args, max_sample=args.num_val)
    dataloader = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=5,
        drop_last=True)

    #builder = DatasetBuilder()
    #dataset = builder.build('cifar10')
    #dataloader = DataLoader(dataset, num_workers=5, drop_last=True, shuffle=False, batch_size=1)
    #print(len(dataloader))

    encoder = ResnetDilated()
    decoder = PPMDeepsup()
    loss = nn.NLLLoss(ignore_index=-1)

    segment = SegmentationModule(encoder, decoder, loss)

    test(segment, dataloader, args)
