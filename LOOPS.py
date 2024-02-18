import torch
import os
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision import transforms
from modelConstruction import yolo_null
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import profile, record_function, ProfilerActivity

from archConfig import decode_bbox
from archConfig import boxFilter
from archConfig import assign_objectness_true
from archConfig import rearr_ab
from archConfig import decode_detecting
from archConfig import normalizeCoor
from archConfig import accuracy
from losses import bboxLos, detectLoss, objsLoss
from tqdm import tqdm
from thop import profile
from push import push_message
from torchvision.ops import nms


import logging

thopLogger = logging.getLogger(__name__)
thopLogger.setLevel(logging.WARNING)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_nulla = yolo_null().to(device)

# evicee = torch.cuda.get_device_name(device)
# print(devicee)

labLibP = '/home/ikase/labs/RSBOT/sources/labelled/yun0Lab/'
srcLibP = '/home/ikase/labs/RSBOT/sources/raw img/yun_0/'

labLib = os.listdir(labLibP)
srcLib = os.listdir(srcLibP)

# anchor boxes
abIs = (46.416, 21.153)
abIIs = (78.584, 60.081)
abIIIs = (108.941, 128.483)
abIVs = (78.584, 60.081)
selfabs = torch.tensor([abIs, abIIs, abIIIs, abIVs]).to(device)

abIm = (157.432, 206.25)
abIIm = (168.374, 34.063)
abIIIm = (195.555, 436.477)
abIVm = (168.374, 34.063)
selfabm = torch.tensor([abIm, abIIm, abIIIm, abIVm]).to(device)

abIx = (223.747, 754.679)
abIIx = (281.654, 67.916)
abIIIx = (449.564, 143.014)
abIVx = (281.654, 67.916)
selfabx = [abIx, abIIx, abIIIx, abIVx]

objectList = {
    0: 'ich',
    1: 'diamantAzg',
    2: 'block',
    3: 'laserwerfer',
    4: 'glasBoden',
    5: 'sprungbrett',
    6: 'bwgBoden',
    7: 'hDrbHammer',
    8: 'flnHammer',
    9: 'fallstelle',
    10: 'Laserpkt',
    11: 'laser',
    12: 'vLaser',
    13: 'vDrbHammer',
    14: 'bwgblock'
}


def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).to(device)


def extBoxes(fileName, path):
    boxDims = []
    file_path = os.path.join(path, fileName)
    scaleW = 1.389
    scaleH = 0.625

    if not fileName.endswith('.xml'):
        return None
    if os.path.isfile(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        for ob in root.findall('object'):
            # name = ob.find('name').text
            bbox = ob.find('bndbox')

            xmin = int(bbox.find('xmin').text) * scaleW
            xmax = int(bbox.find('xmax').text) * scaleW
            ymin = int(bbox.find('ymin').text) * scaleH
            ymax = int(bbox.find('ymax').text) * scaleH

            # boxW = (xmax - xmin)
            # boxH = (ymax - ymin)
            # ctrX = (xmax + xmin) / 2
            # ctrY = (ymax + ymin) / 2

            temp = torch.tensor([xmin, ymin, xmax, ymax])
            boxDims.append(temp)

    boxDims = torch.stack(boxDims)

    return boxDims


def trueClasses(fileName, path):
    classes = []
    file_path = os.path.join(path, fileName)
    if not fileName.endswith('.xml'):
        return None
    if os.path.isfile(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        for ob in root.findall('object'):
            name = ob.find('name').text
            classes.append(name)

    return classes


def trueClassLabels(labels, num_classes, dic):
    one_hot_labels = torch.zeros(len(labels), num_classes).to(int)
    for i, label in enumerate(labels):
        for key, obj in dic.items():
            if obj == label:
                one_hot_labels[i, key] = 1
    return one_hot_labels


def filAnch(trueBox, predBox, predObjs, predClassifi):
    ovlList = []
    ovlObjsList = []
    ovlClassifiList = []
    for BpredBox, BpredObjs, BpredClassifi, BtrueBox in zip(predBox, predObjs, predClassifi, trueBox):
        BBoxList = []
        BObjsList = []
        BClassifiList = []

        for trueBB in BtrueBox:
            trueBB = trueBB.to(device)

            tBoxList = []
            tBClassifiList = []
            for predAB, predObj, predClassif in zip(BpredBox, BpredObjs, BpredClassifi):

                predAB = predAB.to(device)
                predObjsig = torch.sigmoid(predObj).to(device)
                predClassifSft = torch.softmax(predClassif, dim = 0).to(device)

                iou = boxFilter.IoUAT(boxEins = trueBB, boxZwei = predAB)
                if iou > 0.4:
                    tBoxList.append(predAB)
                    BObjsList.append(predObjsig)
                    tBClassifiList.append(predClassifSft)
                else:
                    continue
            if not tBoxList:
                emt = torch.empty(0).to(device)
                BBoxList.append(emt)
                BClassifiList.append(emt)
            else:
                BBoxList.append(torch.stack(tBoxList).to(device))
                BClassifiList.append(torch.stack(tBClassifiList).to(device))

        ovlList.append(BBoxList)
        ovlClassifiList.append(BClassifiList)

        if not BObjsList:
            emp = torch.empty(0).to(device)
            ovlObjsList.append(emp)
        else:
            ovlObjsList.append(torch.stack(BObjsList).to(device))

    return ovlList, ovlObjsList, ovlClassifiList


scaler = torch.cuda.amp.GradScaler()

'''
def training(lr, epochs, dLoader):
    optim = torch.optim.Adam(yolo_nulla.parameters(), lr = lr)
    boundingBoxLoss = bboxLoss()
    objectnessLoss = objsLoss()
    classifiLoss = detectLoss()

    WARNING_TEXT = "\033[91m"
    RESET_TEXT = "\033[0m"
    BOLD_TEXT = "\033[1m"

    prof = True

    yolo_nulla.train()
    totalFails = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for sourceim, trueBoxDims, trueClassKey in tqdm(dLoader, unit = 'batch'):

            with autocast():
                for params in yolo_nulla.parameters():
                    params.requires_grad = True



                try:
                    BOs, COs = yolo_nulla(sourceim)

                    # scale S__________________________________________________________________________
                    BOsDims, BOsObjectness, LFCoors = decode_bbox(BOs, imDim = torch.tensor(800.0, device = device), gridN = torch.tensor(100.0, device = device))
                    BOsDims, BOsObjectness, LFCoors = torch.stack(BOsDims).to(device), torch.stack(BOsObjectness).to(device), torch.stack(LFCoors).to(device)

                    predCOs = decode_detecting(COs)
                    predCOs = torch.stack(predCOs).to(device)

                    predABs = rearr_ab(predBoxDims = BOsDims, ab = selfabs, abCtr = LFCoors)
                    predABs = torch.stack(predABs).to(device)

                    filteredPredABs, filteredPredObjsS, filteredPredClassifS = filAnch(trueBox = trueBoxDims, predBox = predABs, predObjs = BOsObjectness, predClassifi = predCOs)

                    batchLoss = torch.tensor(0.0, device = 'cuda')


                    if prof:
                        flops, totalParams = profile(yolo_nulla, inputs = (sourceim, ))
                        print(f'{BOLD_TEXT} \nbatch loss {batchLoss.item()} \ntotal params {totalParams} \nTFLOPs {flops / 1000000000000} {RESET_TEXT}\n')
                        prof = False
                    else:
                        break

                    for batchABs, batchObjs, batchClassif, batchTrueBoxDims, batchTrueKey in zip(filteredPredABs, filteredPredObjsS, filteredPredClassifS, trueBoxDims, trueClassKey):
                        optim.zero_grad()
                        objectnsLossS = torch.tensor(0.0, device = device, requires_grad = True)
                        bboxLossS = torch.tensor(0.0, device = device, requires_grad = True)
                        classificationLosses_s = torch.tensor(0.0, device = device, requires_grad = True)

                        trueObjsS = assign_objectness_true(predBoxes = batchABs, trueBoxes = batchTrueBoxDims)
                        # losses
                        batchObjs = batchObjs.to(device)

                        if batchObjs.numel() == 0:
                            objectnsLossS = objectnsLossS + torch.tensor(0.0, device = device)
                        else:
                            objectnsLossS = objectnsLossS + objectnessLoss(batchObjs, trueObjsS).to(device)

                        for (trIdx, tr), (trIdxC, trC) in zip(enumerate(batchTrueBoxDims), enumerate(batchTrueKey)):
                            for assoABs, assoClassif in zip(batchABs[trIdx], batchClassif[trIdxC]):
                                if assoABs.numel() == 0:
                                    bboxLossS = bboxLossS + torch.tensor(0.0, device = device)
                                    classificationLosses_s = classificationLosses_s + torch.tensor(0.0, device = device)
                                else:
                                    abb = assoABs.flatten()
                                    abNorm = normalizeCoor(abb)
                                    trNorm = normalizeCoor(tr)
                                    bboxLossS = bboxLossS + boundingBoxLoss(abNorm, trNorm).to(device)

                                    trC = trC.float().to(device)
                                    classificationLosses_s = classificationLosses_s + classifiLoss(assoClassif, trC)

                        totalLossS = bboxLossS + objectnsLossS + classificationLosses_s
                        totalLossS.backward()
                        totalLossS.detach()
                        batchLoss += totalLossS

                    if batchLoss.item() != 0:
                        # scaLoss = scaler.scale(totalLossS)
                        # scaLoss.backward()
                        batchLoss.backward()
                        # scaler.unscale_(optim)
                        # scaler.step(optimizer = optim)
                        optim.step()
                        optim.zero_grad()
                        torch.cuda.empty_cache()
                        # scaler.update()
                        checkpoint = {'model_state_dict': yolo_nulla.state_dict(),
                                      'optimizer_state_dict': optim.state_dict(),
                                      'epoch': epoch,
                                      'batch_loss': batchLoss}
                        torch.save(checkpoint, '/run/media/ikase/LinuSat/weiii/yollo_nulla.pth')
                        print(BOLD_TEXT + 'params updated at checkpoint')
                        print(f'\nbatch loss {batchLoss.item()} {RESET_TEXT}')
                    else:
                        continue

                    with yolo_nulla.inference_mode():
                        randIdx = torch.randint(0, 5, (1, 1))

                        localization_output, classification_output = yolo_nulla(sourceim[randIdx])


                except Exception as e:

                    ovl, rec_code, rec_mes = push_message(message = f'error: {e}')
                    print(f'\n\n{WARNING_TEXT} --- an error message has been sent with status code {rec_code} --- {RESET_TEXT}\n')
                    totalFails += 1

                    if totalFails > 5:
                        push_message(message = 'failed attempts > 5, program interrupted')
                        print('failed attempts > 5, program interrupted')
                        raise SystemExit
                    else:
                        continue
            '''


'''
            # scale M____________________________________________________________________________
            BOmDims, BOmObjectness, LFCoorm = decode_bbox(BOm, imDim = 800, gridN = 50)
            BOmDims, BOmObjectness, LFCoorm = torch.tensor(BOmDims).to(device), torch.tensor(BOmObjectness).to(device), torch.tensor(LFCoorm).to(device)

            predCOm = decode_detecting(COm)
            predCOm = torch.tensor(predCOm, requires_grad = True).to(device)

            predABm = rearr_ab(predBoxDims = BOmDims, ab = selfabm, abCtr = LFCoorm)
            predABm = torch.tensor(predABm, requires_grad = True).to(device)

            filteredPredABm, filteredPredObjsM, filteredPredClassifM = filAnch(trueBox = trueBoxDims, predBox = predABm, predObjs = BOmObjectness, predClassifi = predCOm)

            if filteredPredABm[0] == []:
                totalLossM = torch.tensor(0.0, requires_grad = True, device = device)
            else:
                filteredPredABm, filteredPredObjsM, filteredPredClassifM = torch.stack(filteredPredABm).to(device), torch.stack(filteredPredObjsM).to(device), torch.stack(filteredPredClassifM).to(device)

                trueObjsM = assign_objectness_true(predBoxes = filteredPredABm, trueBoxes = trueBoxDims)
                trueObjsM = trueObjsM.to(device)

                # losses
                objectnsLossM = objectnessLoss(filteredPredObjsM, trueObjsM)

                bboxLossM = 0
                for trIdx, tr in enumerate(trueBoxDims):
                    LfilteredPredABm = len(trueBoxDims) * 2 if len(filteredPredABm) < len(trueBoxDims) else len(filteredPredABm)
                    for batch in filteredPredABm:
                        for ab in batch[int(trIdx * (LfilteredPredABm / len(trueBoxDims))) : int((trIdx + 1) * (LfilteredPredABm / len(trueBoxDims)))]:

                            ab = ab.flatten()
                            abNorm = normalizeCoor(ab)

                            trNorm = normalizeCoor(tr)

                            bboxLossM += boundingBoxLoss(abNorm, trNorm)

                classificationLosses_m = 0
                for trIdx, tr in enumerate(trueClassKey):
                    LfilteredPredClassifM = len(trueClassKey) * 2 if len(filteredPredClassifM) < len(trueClassKey) else len(filteredPredClassifM)
                    for batch in filteredPredClassifM:
                        for predClassifi in batch[int(trIdx * (LfilteredPredClassifM / len(trueClassKey))) : int((trIdx + 1) * (LfilteredPredClassifM / len(trueClassKey)))]:

                            tr = tr.float()

                            classificationLosses_m += classifiLoss(predClassifi, tr)

                totalLossM = bboxLossM + objectnsLossM + classificationLosses_m

            # scale X_________________________________________________________________________
            BOxDims, BOxObjectness, LFCoorx = decode_bbox(BOx, imDim = 800, gridN = 25)
            BOxDims, BOxObjectness, LFCoorx = torch.tensor(BOxDims).to(device), torch.tensor(BOxObjectness).to(device), torch.tensor(LFCoorx).to(device)

            predCOx = decode_detecting(COx)
            predCOx = torch.tensor(predCOx, requires_grad = True).to(device)

            predABx = rearr_ab(predBoxDims = BOxDims, ab = selfabx, abCtr = LFCoorx)
            predABx = torch.tensor(predABx, requires_grad = True).to(device)

            filteredPredABx, filteredPredObjsX, filteredPredClassifX = filAnch(trueBox = trueBoxDims, predBox = predABx, predObjs = BOxObjectness, predClassifi = predCOx)

            if filteredPredABx[0] == []:
                totalLossX = torch.tensor(0.0, requires_grad = True, device = device)
            else:
                filteredPredABx, filteredPredObjsX, filteredPredClassifX = torch.stack(filteredPredABx).to(device), torch.stack(filteredPredObjsX).to(device), torch.stack(filteredPredClassifX).to(device)

                trueObjsX = assign_objectness_true(predBoxes = filteredPredABx, trueBoxes = trueBoxDims)
                trueObjsX = trueObjsX.to(device)

                # losses
                objectnsLossX = objectnessLoss(filteredPredObjsX, trueObjsX)

                bboxLossX = 0
                for trIdx, tr in enumerate(trueBoxDims):
                    LfilteredPredABx = len(trueBoxDims) * 2 if len(filteredPredABx) < len(trueBoxDims) else len(filteredPredABx)
                    for batch in filteredPredABx:
                        for ab in batch[int(trIdx * (LfilteredPredABx / len(trueBoxDims))) : int((trIdx + 1) * (LfilteredPredABx / len(trueBoxDims)))]:

                            ab = ab.flatten()
                            abNorm = normalizeCoor(ab)

                            trNorm = normalizeCoor(tr)

                            bboxLossX += boundingBoxLoss(abNorm, trNorm)

                classificationLosses_x = 0
                for trIdx, tr in enumerate(trueClassKey):
                    LfilteredPredClassifX = len(trueClassKey) * 2 if len(filteredPredClassifX) < len(trueClassKey) else len(filteredPredClassifX)
                    for batch in filteredPredClassifX:
                        for predClassifi in batch[int(trIdx * (LfilteredPredClassifX / len(trueClassKey))) : int((trIdx + 1) * (LfilteredPredClassifX / len(trueClassKey)))]:

                            tr = tr.float()

                            classificationLosses_x += classifiLoss(predClassifi, tr)

                totalLossX = bboxLossX + objectnsLossX + classificationLosses_x

            #
            totalLoss = totalLossS + totalLossM + totalLossX
            epoch_loss += totalLoss.item()
'''


# wooo = training(lr = 0.002, epochs = 20, srcLibb = srcLib, srcLibPP = srcLibP)
