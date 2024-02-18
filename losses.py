from torch import nn


class bboxLos(nn.Module):
    def __init__(self):
        super().__init__()
        self.bxLossFn = nn.MSELoss()

    def forward(self, predBoxes, trueBoxes):
        bxLoss = self.bxLossFn(predBoxes, trueBoxes)

        return bxLoss


class objsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.existLossFn = nn.BCEWithLogitsLoss()

    def forward(self, predExist, trueExist):
        existLoss = self.existLossFn(predExist, trueExist)

        return existLoss


class detectLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.detectLossFn = nn.CrossEntropyLoss()

    def forward(self, predClass, trueClass):
        detectL = self.detectLossFn(predClass, trueClass)

        return detectL
