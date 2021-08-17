from .coco import CocoDataset


class TransDroneDataset(CocoDataset):

    CLASSES = ('Small 1-piece vehicle','Large 1-piece vehicle','Extra-large 2-piece truck','Tractor','Trailer','Motorcycle')

class TransDroneDataset3Cat(TransDroneDataset):

    CLASSES = ('Small 1-piece vehicle','Large 1-piece vehicle','Extra-large 2-piece truck')