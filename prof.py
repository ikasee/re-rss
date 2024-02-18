import torch
from push import push_message
from modelConstruction import yolo_null

device = 'cuda' if torch.cuda.is_available() else push_message(message = 'cuda not available(prof.py)')
if device != 'cuda':
    raise SystemExit('cuda not available')

def ctrCoor_agn(imDim: int, gridN: int):

    samp = imDim / gridN

    ctrCoorXY = torch.linspace(samp / 2, imDim - samp / 2, gridN, device = device)

    xx, yy = torch.meshgrid(ctrCoorXY, ctrCoorXY, indexing = 'xy')
    xx, yy = torch.round(xx), torch.round(yy)

    coords = torch.stack((xx, yy))

    return coords

#ctrCoor_agn(imDim=800, gridN=100)

model = yolo_null().to(device)

def decode_box(bbox_predictions: torch.tensor, ab_n: int):
    ab_groups_dims = []
    ab_groups_objs = []
    for ab_group in range(ab_n):
        ab_group_dims = bbox_predictions[:, ab_group * 5:(ab_group + 1) * 5 - 1, :, :]
        ab_group_objs = bbox_predictions[:, ab_group * 5 + 4:(ab_group + 1) * 5, :, :]
        ab_groups_dims.append(ab_group_dims)
        ab_groups_objs.append(ab_group_objs)

    ab_groups_dims = torch.stack(ab_groups_dims).to(device)
    ab_groups_objs = torch.stack(ab_groups_objs).to(device)

    return ab_groups_dims, ab_groups_objs

def decode_detection(pred_detection: torch.tensor, ab_n: int, class_num: int):
    ab_groups_classifi = []
    for ab_group in range(ab_n):
        ab_group_classifi = pred_detection[:, ab_group * class_num:(ab_group + 1) * class_num, :, :]
        ab_groups_classifi.append(ab_group_classifi)

    ab_groups_classifi = torch.stack(ab_groups_classifi).to(device)

    return ab_groups_classifi

def re_ab(abs: torch.tensor, ab_ctr: torch.tensor, offsets: torch.tensor, batch_size: int, ab_nm):
    abs_reshaped = abs.view(ab_nm, 1, 2, 1, 1).expand(-1, batch_size, -1, 100, 100)
    mul_offsets_scales = offsets[:, :, :2, :, :]
    print(abs_reshaped.shape, mul_offsets_scales.shape)

    newWHs = torch.mul(abs_reshaped, mul_offsets_scales)

    ab_ctr_reshaped = ab_ctr.view(1, 1, 2, 100, 100).expand(ab_nm, batch_size, -1, -1, -1)
    mul_offsets = offsets[:, :, 2:, :, :]
    newCtrs = torch.mul(ab_ctr_reshaped, mul_offsets)

    factor = torch.tensor(2., device = device)

    TLx = torch.sub(newCtrs[:, :, 0, :, :], torch.div(newWHs[:, :, 0, :, :], factor))
    TLy = torch.sub(newCtrs[:, :, 1, :, :], torch.div(newWHs[:, :, 1, :, :], factor))

    RBx = torch.add(newCtrs[:, :, 0, :, :], torch.div(newWHs[:, :, 0, :, :], factor))
    RBy = torch.add(newCtrs[:, :, 1, :, :], torch.div(newWHs[:, :, 1, :, :], factor))

    att = torch.stack((TLx, TLy, RBx, RBy), dim = 2)

    return att

def IoU(predBBoxes: torch.tensor, trueBBoxes: torch.tensor):
    xL = torch.max(trueBBoxes[:, :, 0, :, :], predBBoxes[:, :, 0, :, :])
    yT = torch.max(trueBBoxes[:, :, 1, :, :], predBBoxes[:, :, 1, :, :])
    xR = torch.min(trueBBoxes[:, :, 2, :, :], predBBoxes[:, :, 2, :, :])
    yB = torch.min(trueBBoxes[:, :, 3, :, :], predBBoxes[:, :, 3, :, :])

    wInter = torch.clamp(torch.sub(xR, xL), min = 0)
    hInter = torch.clamp(torch.sub(yB, yT), min = 0)

    AI = torch.mul(wInter, hInter)
    AT = torch.sub(trueBBoxes[:, :, 2, :, :], trueBBoxes[:, :, 0, :, :]) * torch.sub(trueBBoxes[:, :, 3, :, :], trueBBoxes[:, :, 1, :, :])
    AP = torch.sub(predBBoxes[:, :, 2, :, :], predBBoxes[:, :, 0, :, :]) * torch.sub(predBBoxes[:, :, 3, :, :], predBBoxes[:, :, 1, :, :])
    AU = AT + AP - AI

    IoU = AI / AU if torch.any(AU) > 0 and torch.any(AI) > 0 else 0

    return IoU

def filt(trueBBoxes: list[torch.tensor], predBBoxes: torch.tensor, predClassifi: torch.tensor, predObjs: torch.tensor, classes_n: int, ab_n: int, grid_n: int):
    ovl = []
    ovlObjs = []
    ovlTObjs = []
    ovlBBox = []
    ovlClassifi = []
    for btbInd, trueBoxes in enumerate(trueBBoxes):
        rsTrueBoxes = trueBoxes.view(int(trueBoxes.size(0)), 1, 4, 1, 1).expand(-1, ab_n, -1, grid_n, grid_n)

        rsPredBBoxes = predBBoxes[:, btbInd, :, :, :]
        rsPredBBoxes = rsPredBBoxes.view(1, ab_n, 4, grid_n, grid_n).expand(int(trueBoxes.size(0)), -1, -1, -1, -1)
        ovlBBox.append(rsPredBBoxes)

        rsPredClassifi = predClassifi[:, btbInd, :, :, :]
        rsPredClassifi = rsPredClassifi.view(1, ab_n, classes_n, grid_n, grid_n).expand(int(trueBoxes.size(0)), -1, -1, -1, -1)
        ovlClassifi.append(rsPredClassifi)

        rsPredObjs = predObjs[:, btbInd, :, :, :]
        rsPredObjs = rsPredObjs.view(1, ab_n, 1, grid_n, grid_n).expand(int(trueBoxes.size(0)), -1, -1, -1, -1)

        abIoU = IoU(predBBoxes = rsPredBBoxes, trueBBoxes = rsTrueBoxes)

        IoU_mask = (abIoU > 0.001)
        IoU_mask = IoU_mask if isinstance(IoU_mask, torch.Tensor) else torch.tensor(IoU_mask, device = device)
        fild_Ind = torch.nonzero(IoU_mask)

        if fild_Ind.numel() == 0:
            ovlTObjs.append(torch.zeros((1, 1), device = device, dtype = torch.float16))
            ovlObjs.append(torch.zeros((1, 1), device = device, dtype = torch.float16))
            ovl.append(torch.tensor([([0, torch.randint(low = 0, high = 3, size = (1, 1)).item(), torch.randint(low = 0, high = 99, size = (1, 1)).item(), torch.randint(low = 0, high = 99, size = (1, 1)).item()])], device = device))
        else:
            d0 = fild_Ind[:, 0]
            d1 = fild_Ind[:, 1]
            d3 = fild_Ind[:, 2]
            d4 = fild_Ind[:, 3]

            objsFild = rsPredObjs[d0, d1, :, d3, d4]

            tObjs_a = torch.zeros((objsFild.size(0), 1), device = device)
            tObjs_mask_c = abIoU[IoU_mask]
            for ind, iou in enumerate(tObjs_mask_c):
                if iou > 0.07:
                    tObjs_a[ind, 0] = 1
                else:
                    continue

            ovlTObjs.append(tObjs_a)
            ovlObjs.append(objsFild)
            ovl.append(fild_Ind)

    return ovl, ovlObjs, ovlTObjs, ovlBBox, ovlClassifi

def normCoor(boxes: torch.tensor, imageWidth: int, imageHeight: int):
    xmin = boxes[:, 0] / imageWidth
    ymin = boxes[:, 1] / imageHeight
    xmax = boxes[:, 2] / imageWidth
    ymax = boxes[:, 3] / imageHeight

    ovl = torch.stack((xmin, ymin, xmax, ymax), dim = 1)

    return ovl

