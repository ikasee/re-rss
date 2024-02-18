import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import matplotlib.pyplot as mtpt
from archConfig import modConv
from archConfig import SPPF
from archConfig import CSP
from archConfig import classificationProcess


class yolo_null(nn.Module):
    def __init__(self):
        super().__init__()
        # h
        num_classes = 15
        num_a = 4
        # dropout
        self.dropout = nn.Dropout(p = 0.2)

        # softmax
        # self.softmax = nn.Softmax()

        # upsample
        self.upsample = nn.Upsample(scale_factor= (2, 2), mode = 'bilinear', align_corners = True)

        # stem layer
        self.stemLI = modConv(convKSize = 3, convStride = 2, convPad = 1, convInCh = 3, convOutCh = 64)

        # stage layers
        self.stageILI = modConv(convKSize = 3, convStride = 2, convPad = 1, convInCh = 64, convOutCh = 128)
        self.stageILII = CSP(CSPSkipConn = True, nLayers = 3, CSPInCh = 128)

        self.stageIILI = modConv(convKSize = 3, convStride = 2, convPad = 1, convInCh = 128, convOutCh = 256)
        self.stageIILII = CSP(CSPSkipConn = True, nLayers = 6, CSPInCh = 256)

        self.stageIIILI = modConv(convKSize = 3, convStride = 2, convPad = 1, convInCh = 256, convOutCh = 512)
        self.stageIIILII = CSP(CSPSkipConn = True, nLayers = 6, CSPInCh = 512)

        self.stageIVLI = modConv(convKSize = 3, convStride = 2, convPad = 1, convInCh = 512, convOutCh = 512)
        self.stageIVLII = CSP(CSPSkipConn = True, nLayers = 3, CSPInCh = 512)
        self.stageIVLIII = SPPF(sppfIn = 512, sppfOut = 512)

        # top down layers
        self.TDLIICSP = CSP(CSPSkipConn = False, nLayers = 3, CSPInCh = 1024)

        self.TDLICSP = CSP(CSPSkipConn = False, nLayers = 3, CSPInCh = 768)

        # output layers
        self.headOIBConvMod = modConv(convKSize = 3, convStride = 1, convPad = 1, convInCh = 256, convOutCh = 256)
        self.headOIBConv = nn.Conv2d(kernel_size = 1, stride = 1, padding = 0, in_channels = 256, out_channels = (4 + 1) * num_a)
        self.headOICConvMod = modConv(convKSize = 3, convStride = 1, convPad = 1, convInCh = 256, convOutCh = 256)
        self.headOICConv = nn.Conv2d(kernel_size = 1, stride = 1, padding = 0, in_channels = 256, out_channels = num_classes * num_a)

        # self.headOIIIBConvMod = modConv(convKSize = 3, convStride = 1, convPad = 1, convInCh = 512, convOutCh = 512)
        # self.headOIIIBConv = nn.Conv2d(kernel_size = 1, stride = 1, padding = 0, in_channels = 512, out_channels = (4 + 1) * num_a)
        # self.headOIIICConvMod = modConv(convKSize = 3, convStride = 1, convPad = 1, convInCh = 512, convOutCh = 512)
        # self.headOIIICConv = nn.Conv2d(kernel_size = 1, stride = 1, padding = 0, in_channels = 512, out_channels = num_classes * num_a)

        # DSBU
        # self.DSBUIConv = modConv(convKSize = 3, convStride = 2, convPad = 1, convInCh = 256, convOutCh = 256)
        # self.DSBUICSP = CSP(CSPSkipConn = False, nLayers = 3, CSPInCh = 768)

        # self.DSBUIIConv = modConv(convKSize = 3, convStride = 2, convPad = 1, convInCh = 512, convOutCh = 512)
        # self.DSBUIICSP = CSP(CSPSkipConn = False, nLayers = 3, CSPInCh = 1024)

        # channel downsampling
        self. sampleDownConv = nn.Conv2d(kernel_size = 1, stride = 1, in_channels = 1024, out_channels = 512)
        self. sampleDownConvI = nn.Conv2d(kernel_size = 1, stride = 1, in_channels = 768, out_channels = 256)
        # self.sampleDownConvII = nn.Conv2d(kernel_size = 1, stride = 1, in_channels = 768, out_channels = 512)
        # self.sampleDownConvIII = nn.Conv2d(kernel_size = 1, stride = 1, in_channels = 1024, out_channels = 512)

    def forward(self, a):
        # Backbone outputs
        outN = self.stemLI(a)
        outN1 = self.stageIILII(self.dropout(self.stageIILI(self.stageILII(self.stageILI(outN)))))
        outN2 = self.stageIIILII(self.dropout(self.stageIIILI(outN1)))
        outN3 = self.stageIVLIII(self.dropout(self.stageIVLII(self.stageIVLI(outN2))))

        # topdown
        TDLN2Cat = torch.cat([self.upsample(outN3), outN2], dim=1)
        TDLN2 = self.sampleDownConv(self.TDLIICSP(TDLN2Cat))

        TDLN1Cat = torch.cat([self.upsample(TDLN2), outN1], dim=1)
        TDLN1 = self.sampleDownConvI(self.TDLICSP(TDLN1Cat))

        # output I
        foIB = self.headOIBConv(self.headOIBConvMod(TDLN1))
        foIC = self.headOICConv(self.headOICConvMod(TDLN1))

        '''
        # DSBUI
        DSBUIOut = self.DSBUICSP(torch.cat([self.DSBUIConv(TDLN1), TDLN2], dim = 1))
        DSBUIOut = self.sampleDownConvII(DSBUIOut)

        # output II
        foIIB = self.headOIIIBConv(self.headOIIIBConvMod(DSBUIOut))
        foIIC = self.headOIIICConv(self.headOIIICConvMod(DSBUIOut))

        # DSBUII
        DSBUIIOut = self.DSBUIICSP(torch.cat([outN3, self.DSBUIIConv(DSBUIOut)], dim = 1))
        DSBUIIOut = self.sampleDownConvIII(DSBUIIOut)

        # output III
        foIIIB = self.headOIIIBConv(self.headOIIIBConvMod(DSBUIIOut))
        foIIIC = self.headOIIICConv(self.headOIIICConvMod(DSBUIIOut))
        '''
        return foIB, foIC  # , foIIB, foIIC, foIIIB, foIIIC
