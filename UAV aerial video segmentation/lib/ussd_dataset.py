import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import numpy as np

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
from lib.base_dataset import BaseDataset, TransformationTrain, TransformationVal


class USSD(BaseDataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(USSD, self).__init__(dataroot, annpath, trans_func, mode)

        self.n_cats = 6         # 类别数 （实际上在本脚本中并没有用到过）
        self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)

        # for el in labels_info:
        #     self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.5391, 0.6011, 0.6022),
            std=(0.1888, 0.1599, 0.1751),
        )


def get_data_loader(dataroot, annpath, ims_per_gpu, scales, cropsize, max_iter=None, mode='train', distributed=True):
    if mode == 'train':
        trans_func = TransformationTrain(scales, cropsize)
        batchsize = ims_per_gpu
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = TransformationVal()
        batchsize = ims_per_gpu
        shuffle = False
        drop_last = False

    ds = USSD(dataroot, annpath, trans_func=trans_func, mode=mode)

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not max_iter is None
            n_train_imgs = ims_per_gpu * dist.get_world_size() * max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
        )
    return dl


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # ds = Potsdam('./data/', mode='train')
    ds = USSD(dataroot=r'F:\MachineLearning-Datasets\Potsdam',
              annpath=r'F:\ZYHWorkSpace\BiSeNet\datasets\Potsdam\train_win.txt',
              mode='train')
    dl = DataLoader(ds,
                    batch_size=4,
                    shuffle=True,
                    num_workers=4,
                    drop_last=True)

    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
