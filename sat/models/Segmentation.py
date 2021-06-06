import torch.nn as nn


class SegmentationModule(nn.Module):
    """
    Semantic segmentation module used by ColorFool to perturb different semantic regions.
    """
    def __init__(self, net_enc, net_dec, crit):
        """

        :param net_enc: Segmentation encoder.
        :param net_dec:  Segmentation decoder.
        :param crit: Loss function of the module.
        """
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit

    def forward(self, data, segSize):
        return self.decoder(self.encoder(data, return_feature_maps=True), segSize=segSize)
