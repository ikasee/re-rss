import torch
from torch import nn
from torch.nn.modules.pooling import MaxPool2d
from itertools import combinations

# defW, defH, defC = None, None, None
# defIn, defOut = None, None


class modConv(nn.Module):
    def __init__(self, convKSize, convStride, convPad, convInCh, convOutCh):
        super(modConv, self).__init__()

        convInCh = int(convInCh)
        convOutCh = int(convOutCh)

        self.layConv = nn.Conv2d(in_channels = convInCh, out_channels = convOutCh, kernel_size = convKSize, stride = convStride, padding = convPad, bias = False)
        self.layNorm = nn.BatchNorm2d(num_features = convOutCh)
        self.layAct = nn.SiLU()

    def forward(self, a):
        return self.layAct(self.layNorm(self.layConv(a)))


class darknet(nn.Module):
    def __init__(self, dkSkipConn, dkInCh: int, dkReduc = 0.5, dkKSize = 3, dkStride = 1, dkPad = 1):
        super(darknet, self).__init__()

        if dkInCh is not None:
            ch_reduc = dkInCh * dkReduc
        else:
            raise SystemError('dkInCh cannot be None')
        
        self.dkConvI = modConv(convInCh = dkInCh, convOutCh = ch_reduc, convKSize = dkKSize, convStride = dkStride, convPad = dkPad)
        self.dkConvII = modConv(convInCh = ch_reduc, convOutCh = dkInCh, convKSize = dkKSize, convStride = dkStride, convPad = dkPad)
        self.dkSkipConn = dkSkipConn

    def forward(self, a):
        ident = a if self.dkSkipConn else None
        out = self.dkConvI(a)
        out = self.dkConvII(out)

        if self.dkSkipConn:
            out += ident

        return out


class CSP(nn.Module):
    def __init__(self, nLayers, CSPSkipConn, CSPInCh):
        super(CSP, self).__init__()

        self.nLayers = nLayers
        self.CSPInCh = CSPInCh

        self.cspConvI = modConv(convInCh = CSPInCh, convOutCh = CSPInCh * 0.5, convKSize = 1, convStride = 1, convPad = 0)
        self.cspConvII = modConv(convInCh = int(0.5 * (nLayers + 2) * CSPInCh), convOutCh = CSPInCh, convKSize = 1, convStride = 1, convPad = 0)
        self.cspBottleneckLayers = self.cspBottleneckLayers = nn.ModuleList([darknet(dkSkipConn = CSPSkipConn, dkInCh = CSPInCh * 0.5) for _ in range(nLayers)])
        self.cspConvIII = modConv(convInCh = 4 * (0.5 * CSPInCh), convOutCh = int(0.5 * (self.nLayers + 2) * self.CSPInCh), convKSize = 1, convStride = 1, convPad = 0)

    def forward(self, a):
        cspConvIOut = self.cspConvI(a)
        # convRandInd = torch.randint(low = 0, high = cspConvIOut.size(1), size = (int(cspConvIOut.size(1) * 0.5),))
        
        splitI = cspConvIOut
        splitII = cspConvIOut
        splitdkIn = cspConvIOut

        dkbnIOut = self.cspBottleneckLayers[0](splitdkIn)
        # dkbnISplitInd = torch.randint(low = 0, high = dkbnIOut.size(1), size = (int(dkbnIOut.size(1) * 0.5),))
        dkbnIOutI = dkbnIOut
        dkbnIOutII = dkbnIOut

        for dkLayers in self.cspBottleneckLayers[0:]:
            dkbnIOutII = dkLayers(dkbnIOutII)

        cspConcat = torch.cat((splitI, splitII, dkbnIOutI, dkbnIOutII), 1)

        out = self.cspConvII(self.cspConvIII(cspConcat))

        return out


class SPPF(nn.Module):
    def __init__(self, sppfIn, sppfOut):
        super(SPPF, self).__init__()
        
        self.convLayerI = modConv(convInCh = sppfIn, convOutCh = sppfOut, convKSize = 1, convStride = 1, convPad = 0)
        self.convLayerII = modConv(convInCh = sppfIn * 4, convOutCh = sppfOut, convKSize = 1, convStride = 1, convPad = 0)
        self.mpoolI = MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.mpoolII = MaxPool2d(kernel_size = 5, stride = 1, padding = 2)
        self.mpoolIII = MaxPool2d(kernel_size = 7, stride = 1, padding = 3)

    def forward(self, a):
        splita = []
        convI = self.convLayerI(a)
        splita.append(convI)
        mpI = self.mpoolI(convI)
        splita.append(mpI)
        mpII = self.mpoolII(mpI)
        splita.append(mpII)
        mpIII = self.mpoolIII(mpII)
        concatLayer = torch.cat((*splita, mpIII), dim = 1)
        convII = self.convLayerII(concatLayer)

        return convII


# blocks below are substituted by correspond blocks in prof.py
def classificationProcess(classPreds):
    probs = torch.sigmoid(classPreds)
    classification = torch.argmax(probs, dim = -1)

    return probs, classification


def decode_bbox(boundingBoxPredictions, imDim, gridN):
    ab_n = 4
    bbp = boundingBoxPredictions
    predBxList = []
    predObjsList = []
    ctrCoorList = []

    for batch in range(bbp.size(0)):
        BpredBxList = []
        BpredObjsList = []
        BctrCoorList = []
        for gridX in range(bbp.size(2)):
            for gridY in range(bbp.size(3)):
                apartDims = []

                for abInd in range(ab_n):
                    dimList = bbp[batch, abInd * 5 : (abInd + 1) * 4, gridX, gridY].tolist()
                    predObjs = bbp[batch, abInd * 5 + 4, gridX, gridY]

                    objsList = [predObjs]
                    objsList = torch.tensor(objsList, device = 'cuda')

                    apartDims.append(torch.tensor(dimList, device = 'cuda'))
                    BpredObjsList.append(objsList)

                BpredBxList.append(torch.stack(apartDims).to('cuda'))

                gridWH = imDim / gridN
                ctrCoor = [0 + (gridY * gridWH) + (gridWH / 2), 0 + (gridX * gridWH) + (gridWH / 2)]
                BctrCoorList.append(torch.tensor(ctrCoor, device = 'cuda'))
        predBxList.append(torch.stack(BpredBxList).to('cuda'))
        predObjsList.append(torch.stack(BpredObjsList).to('cuda'))
        ctrCoorList.append(torch.stack(BctrCoorList).to('cuda'))

    return predBxList, predObjsList, ctrCoorList


def decode_detecting(classificationPredictions, nClass = 15, nAB = 4):
    ab_n = 4
    cfp = classificationPredictions
    predDtcList = []
    for batch in range(cfp.size(0)):
        BpredDtcList = []
        for gridX in range(cfp.size(2)):
            for gridY in range(cfp.size(3)):
                for abInd in range(ab_n):
                    dtcList = cfp[batch, abInd * nClass : (abInd + 1) * nClass, gridX, gridY].tolist()
                    BpredDtcList.append(torch.tensor(dtcList, device = 'cuda'))
        predDtcList.append(torch.stack(BpredDtcList).to('cuda'))

    return predDtcList


def rearr_ab(predBoxDims, ab, abCtr):
    ovlCoor = []
    for bIdx, batch in enumerate(predBoxDims):
        BovlCoor = []
        BabCtr = abCtr[bIdx]

        for cellIdx, cell in enumerate(batch):
            cabCtr = BabCtr[cellIdx]
            for idx, predBox in enumerate(cell):

                offsetX = predBox[2]
                offsetY = predBox[3]
                scaleW = predBox[0]
                scaleH = predBox[1]

                altT = torch.tensor((10001, 10001), device = 'cuda')

                cab = ab[idx]
                O, I = torch.where(cabCtr < 10000, cabCtr, altT).to('cuda')

                newX = offsetX + O
                newY = offsetY + I

                newW = scaleW * cab[0]
                newH = scaleH * cab[1]

                lt = (newX - newW / 2, newY - newH / 2)
                rb = (newX + newW / 2, newY + newH / 2)
                coorList = [lt, rb]

                BovlCoor.append(torch.tensor(coorList, device = 'cuda'))
        ovlCoor.append(torch.stack(BovlCoor).to('cuda'))
    return ovlCoor


class boxFilter:
    @staticmethod
    def prediction(boxes, bBClassPreds, threshold):
        classPreds, classScores = torch.max(bBClassPreds, dim = 1)
        mask = classScores >= threshold

        bB = boxes[mask]
        bBPreds = classPreds[mask]
        bBScores = classScores[mask]

        return bB, bBPreds, bBScores

    @staticmethod
    def IoU(boxEins, boxZwei):
        boxEins = boxEins.to('cuda')
        boxZwei = boxZwei.to('cuda')
        idk = torch.tensor(0.0, device = 'cuda')

        xL = torch.max(boxEins[0][0], boxZwei[0])
        yT = torch.max(boxEins[0][1], boxZwei[1])
        xR = torch.min(boxEins[1][0], boxZwei[2])
        yB = torch.min(boxEins[1][1], boxZwei[3])
        wInter = torch.max(idk, xR - xL)
        hInter = torch.max(idk, yB - yT)

        Ainter = torch.mul(wInter, hInter)

        AboxEins = (boxEins[1][0] - boxEins[0][0]) * (boxEins[1][1] - boxEins[0][1])
        AboxZwei = (boxZwei[2] - boxZwei[0]) * (boxZwei[3] - boxZwei[1])
        Aunite = AboxEins + AboxZwei - Ainter

        IoU = (Ainter / Aunite) if Ainter > 0 else 0

        return IoU
    count = 0
    @staticmethod
    def IoUAT(boxEins, boxZwei):
        idk = torch.tensor(0.0, device = 'cuda')
        xL = torch.max(boxEins[0], boxZwei[0][0])
        yT = torch.max(boxEins[1], boxZwei[0][1])
        xR = torch.min(boxEins[2], boxZwei[1][0])
        yB = torch.min(boxEins[3], boxZwei[1][1])

        wInter = torch.max(idk, xR - xL)
        hInter = torch.max(idk, yB - yT)

        Ainter = torch.mul(wInter, hInter)

        AboxEins = (boxEins[2] - boxEins[0]) * (boxEins[3] - boxEins[1])
        AboxZwei = (boxZwei[1][0] - boxZwei[0][0]) * (boxZwei[1][1] - boxZwei[0][1])
        Aunite = AboxEins + AboxZwei - Ainter

        IoU = (Ainter / Aunite) if Ainter > 0 else 0
        return IoU

def corner_to_corner(box):
    W = box[0]
    H = box[1]
    X = box[2]
    Y = box[3]
    lt = (X - (W / 2), Y - (H / 2))
    rb = (X + (W / 2), Y + (H / 2))
    ltrb = lt, rb
    lis = torch.tensor(ltrb)
    return lis


def assign_objectness_true(predBoxes, trueBoxes, iou_threshold=0.5):

    flatPredBoxes = []
    for tens in predBoxes:
        flatPredBoxes.extend(tens.tolist())
    flatPredBoxes = torch.tensor(flatPredBoxes)

    predsLen = flatPredBoxes.size(0)
    labels = torch.zeros((predsLen, 1), device = 'cuda')
    for bxIdx, bx in enumerate(flatPredBoxes):
        for trueBox in trueBoxes:
            iou = boxFilter.IoU(bx, trueBox)
            if iou >= iou_threshold:
                labels[bxIdx, 0] = 1

    return labels

def normalizeCoor(box, imageWidth = 800, imageHeight = 800):
    xmin = box[0] / imageWidth
    ymin = box[1] / imageHeight
    xmax = box[2] / imageWidth
    ymax = box[3] / imageHeight
    lis = [xmin, ymin, xmax, ymax]
    liss = torch.tensor(lis).to('cuda')

    return liss

import torch

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, dim=1)

    total = labels.size(0)


    correct = (predicted == labels).sum().item()


    accuracy = (correct / total) * 100.0
    return accuracy
