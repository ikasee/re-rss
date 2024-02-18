import torch
from push import push_message
from prof import ctrCoor_agn, decode_box, re_ab, decode_detection, filt, normCoor
from losses import bboxLos, objsLoss, detectLoss
from modelConstruction import yolo_null
from tqdm import tqdm
from torch.cuda.amp import grad_scaler, autocast
from thop import profile
import torch.optim.lr_scheduler as lr_scheduler
import inspect
import traceback
import matplotlib.pyplot as mtpt
from mpl_toolkits.mplot3d import Axes3D

# from torch.utils.tensorboard import SummaryWriter

WARNING_TEXT = "\033[31m"
RESET_TEXT = "\033[0m"
BOLD_TEXT = "\033[92m"
PURP_TEXT = '\033[95m'

device = 'cuda' if torch.cuda.is_available() else push_message(message=f'cuda not available(LOOPSII)')
if device != 'cuda':
    raise SystemExit('cuda not available')

yolo_nulla = yolo_null().to(device)


def loops(lr, epochs, dLoader, valLoader):
    # boardLog = SummaryWriter('/home/ikase/labs/RSBOT/raws/')
    loss_list = []
    batch_n_list = []
    epoch_n_list = []

    optim = torch.optim.Adam(yolo_nulla.parameters(), lr=lr)

    boundingBoxLoss = bboxLos()
    objectnessLoss = objsLoss()
    classifiLoss = detectLoss()

    prof = True

    yn_path = '/run/media/ikase/LinuSat/weiii/yn/yolo_mod.pth'
    optim_path = '/run/media/ikase/LinuSat/weiii/yn/yolo_optim.pth'
    sche_path = '/run/media/ikase/LinuSat/weiii/yn/yolo_sche.pth'

    yn_stats = torch.load(yn_path)
    yolo_nulla.load_state_dict(yn_stats, strict=False, assign=True)
    optim_stats = torch.load(optim_path)
    optim.load_state_dict(optim_stats)

    total_fails = 0

    abIs = torch.tensor((46.416, 21.153), device=device)
    abIIs = torch.tensor((78.584, 60.081), device=device)
    abIIIs = torch.tensor((108.941, 128.483), device=device)
    abIVs = torch.tensor((78.584, 60.081), device=device)
    abWHs = torch.stack([abIs, abIIs, abIIIs, abIVs], dim=0)

    ctrCoords = ctrCoor_agn(800, 100)

    scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode = 'min', patience = 4, factor = 0.1, threshold = 0.0001)

    sche_stats = torch.load(sche_path)
    scheduler.load_state_dict(sche_stats)

    best_val_loss = float(100)

    for epoch in range(epochs):
        epoch_loss = 0
        batch_n = 0
        avgBLoss = 0
        #try:
        yolo_nulla.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            print('validating')
            for batchidx, (sourceim, trueBoxDims, trueClassKey) in enumerate(valLoader):
                with autocast():
                    BOs, COs = yolo_nulla(sourceim)
                    # B = transforms.ToPILImage()
                    # BB = B(sourceim[0])
                    # BB.show()

                    pred_offsets, pred_objs = decode_box(bbox_predictions=BOs, ab_n=4)

                    predABs = re_ab(abs=abWHs, ab_ctr=ctrCoords, offsets=pred_offsets, batch_size=2, ab_nm=4)

                    predClassifi = decode_detection(pred_detection=COs, ab_n=4, class_num=15)

                    fildInd, fildObjs, trueObjs, hFildBBox, hFildClassifi = filt(trueBoxDims, predABs, predClassifi,
                                                                                 pred_objs, classes_n=15, ab_n=4,
                                                                                 grid_n=100)

                    # ___________________________________________________________________________________________________

                    batchValLoss = torch.tensor(0., device=device, requires_grad=True)

                    for bInd in range(sourceim.size(0)):

                        objectnsLoss = torch.tensor(0.0, device=device, requires_grad=True)
                        bboxLoss = torch.tensor(0.0, device=device, requires_grad=True)
                        classificationLoss = torch.tensor(0.0, device=device, requires_grad=True)

                        objectnsLoss = objectnsLoss + objectnessLoss(fildObjs[bInd], trueObjs[bInd])

                        bFildInd = fildInd[bInd]
                        uqBtbIndd = torch.unique(bFildInd[:, 0])
                        for uqBtbInd in uqBtbIndd:
                            condition = bFildInd[:, 0] == int(uqBtbInd)
                            apr = torch.sum(condition)
                            aprInd = bFildInd[condition]

                            d0 = aprInd[:, 0]
                            d1 = aprInd[:, 1]
                            d3 = aprInd[:, 2]
                            d4 = aprInd[:, 3]

                            # ________________________________________________
                            fildBoxes = hFildBBox[bInd][d0, d1, :, d3, d4]

                            trueBoxSi = trueBoxDims[bInd][uqBtbInd]
                            trueBoxMu = trueBoxSi.view(1, 4).expand(apr, -1)

                            normdPredB, normdTrueB = normCoor(fildBoxes, 800, 800), normCoor(trueBoxMu, 800, 800)

                            bboxLoss = bboxLoss + boundingBoxLoss(normdPredB, normdTrueB)

                            fildClassifi = hFildClassifi[bInd][d0, d1, :, d3, d4]

                            trueClassifiSi = trueClassKey[bInd][uqBtbInd].to(torch.float16)
                            trueClassifiMu = trueClassifiSi.view(1, 15).expand(apr, -1)

                            classificationLoss = classificationLoss + classifiLoss(fildClassifi, trueClassifiMu)
                        batchValLoss = batchValLoss + objectnsLoss + bboxLoss + classificationLoss
                        epoch_val_loss +=batchValLoss.item()

        avg_epoch_val_loss = epoch_val_loss / len(valLoader)

        for batchidx, (sourceim, trueBoxDims, trueClassKey) in enumerate(tqdm(dLoader, unit='batch')):
            yolo_nulla.train()
            optim.zero_grad()
            batch_n += 1
            with autocast():
                for params in yolo_nulla.parameters():
                    params.requires_grad = True

                BOs, COs = yolo_nulla(sourceim)

                # B = transforms.ToPILImage()
                # BB = B(sourceim[0])
                # BB.show()

                pred_offsets, pred_objs = decode_box(bbox_predictions=BOs, ab_n=4)

                predABs = re_ab(abs=abWHs, ab_ctr=ctrCoords, offsets=pred_offsets, batch_size=4, ab_nm=4)

                predClassifi = decode_detection(pred_detection=COs, ab_n=4, class_num=15)

                fildInd, fildObjs, trueObjs, hFildBBox, hFildClassifi = filt(trueBoxDims, predABs, predClassifi,
                                                                             pred_objs, classes_n=15, ab_n=4,
                                                                             grid_n=100)

                # ___________________________________________________________________________________________________

                batchLoss = torch.tensor(0., device=device, requires_grad=True)
                o = 0
                if prof:
                    flops, totalParams = profile(yolo_nulla, inputs=(sourceim,))
                    totalParams = f'{totalParams:,}'
                    print(f' \n{PURP_TEXT}total params {totalParams} \nTFLOPs {flops / 1e12} {RESET_TEXT} \n')
                    prof = False

                for bInd in range(sourceim.size(0)):

                    objectnsLoss = torch.tensor(0.0, device=device, requires_grad=True)
                    bboxLoss = torch.tensor(0.0, device=device, requires_grad=True)
                    classificationLoss = torch.tensor(0.0, device=device, requires_grad=True)

                    objectnsLoss = objectnsLoss + objectnessLoss(fildObjs[bInd], trueObjs[bInd])

                    bFildInd = fildInd[bInd]
                    uqBtbIndd = torch.unique(bFildInd[:, 0])
                    for uqBtbInd in uqBtbIndd:
                        condition = bFildInd[:, 0] == int(uqBtbInd)
                        apr = torch.sum(condition)
                        aprInd = bFildInd[condition]

                        d0 = aprInd[:, 0]
                        d1 = aprInd[:, 1]
                        d3 = aprInd[:, 2]
                        d4 = aprInd[:, 3]

                        # ________________________________________________
                        fildBoxes = hFildBBox[bInd][d0, d1, :, d3, d4]

                        trueBoxSi = trueBoxDims[bInd][uqBtbInd]
                        trueBoxMu = trueBoxSi.view(1, 4).expand(apr, -1)

                        normdPredB, normdTrueB = normCoor(fildBoxes, 800, 800), normCoor(trueBoxMu, 800, 800)

                        bboxLoss = bboxLoss + boundingBoxLoss(normdPredB, normdTrueB)

                        fildClassifi = hFildClassifi[bInd][d0, d1, :, d3, d4]

                        trueClassifiSi = trueClassKey[bInd][uqBtbInd].to(torch.float16)
                        trueClassifiMu = trueClassifiSi.view(1, 15).expand(apr, -1)

                        classificationLoss = classificationLoss + classifiLoss(fildClassifi, trueClassifiMu)
                    batchLoss = batchLoss + objectnsLoss + bboxLoss + classificationLoss
                    avgBLoss += batchLoss.detach().item()

                batchLoss.backward()
                optim.step()
                # scaLoss = scaler.scale(totalLossS)
                # scaLoss.backward()
                # scaler.unscale_(optim)
                # scaler.step(optimizer = optim)
                # scaler.update()
                if batchidx % 20 == 0 or batchidx == 516:
                    loss_list.append(avgBLoss)
                    batch_n_list.append(batch_n)
                    epoch_n_list.append(epoch)
                # boardLog.add_scalar('average training loss per 20 batch', avgBatchLoss, batchidx)
                    avgBaLoss = 0

                epoch_loss += batchLoss



        '''
            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                with open(yn_path, 'wb') as f:
                    yn_state_dict = yolo_nulla.state_dict()
                    yn_state_dict.pop('total_ops', None)
                    yn_state_dict.pop('total_params', None)
                    torch.save(yn_state_dict, f=f)

                with open(optim_path, 'wb') as f:
                    optim_state_dict = optim.state_dict()
                    torch.save(optim_state_dict, f=f)

                with open(sche_path, 'wb') as f:
                    scheduler_state_dict = scheduler.state_dict()
                    torch.save(scheduler_state_dict, f=f)
                    print(f'{BOLD_TEXT} \nbatch {batchidx}: \n-params updated at checkpoint-')
                    print(f'-batch loss {batchLoss}- \n-batch val loss {epoch_val_loss / batch_n}- {RESET_TEXT}\n')

            scheduler.step(avg_epoch_val_loss)

        except Exception as exc:
            tb = traceback.extract_tb(exc.__traceback__)
            filename, line_number, func_name, text = tb[-1]
            ovl, rec_code, rec_mes = push_message(message=f'ERROR-- \n{filename}, {line_number} \n\n{text} \n\n*--{exc}*')
            print(f'\n{WARNING_TEXT} --- an error message has been sent with status code {rec_code} --- {RESET_TEXT}\n')
            total_fails += 1
            if total_fails > 4:
                push_message(message='__failed attempts > 5, program interrupted__')
                raise SystemExit(f'\n{BOLD_TEXT}failed attempts > 5, program interrupted{RESET_TEXT}\n')
            else:
                continue

        push_message(message=f'epoch {epoch } loss per batch: {epoch_loss / batch_n}')
        print(f'epoch {epoch} average batch loss: {epoch_loss / batch_n}')
        if epoch_loss / batch_n < 6:
            raise SystemExit

    fig = mtpt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    X = [epoch for epoch in epoch_n_list]
    Y = [batch for batch in batch_n_list]
    Z = [losses for losses in loss_list]

    ax.scatter(X, Y, Z, c=Z, cmap='viridis')

    ax.set_xlabel('epoch number')
    ax.set_ylabel('batch number')
    ax.set_zlabel('average batch loss per 20 batches')

    ax.set_title('yn_loss_plot')

    mtpt.show()

    push_message(message=f'__{epochs} epochs executed__')
    '''