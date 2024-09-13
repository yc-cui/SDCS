from torch.utils.data import Dataset
import tifffile
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from kornia.augmentation import RandomCrop
from torchvision.transforms import functional as F


def _slide(x, k):
    B, C, H, W = x.shape
    patches = x.unfold(2, k, k).unfold(3, k, k).permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C, k, k)
    return patches


class SlideDataset(Dataset):
    def __init__(self, pairs, img_size, slide_size, max_val, train=True):
        self.pairs = pairs
        self.slide_size = slide_size
        self.img_size = img_size
        self.train = train
        
        LRs_arr = []
        HRs_arr = []
        CSBS_arr = []
        for LR, HR, CSBS in zip(pairs['LRs'], pairs['HRs'], pairs["CSBS"]):
            LR_data = np.array(tifffile.imread(LR))
            HR_data = np.array(tifffile.imread(HR))
            CSBS_data = np.array(tifffile.imread(CSBS))
            LR_data[LR_data < 0] = 0
            HR_data[HR_data < 0] = 0
            CSBS_data[CSBS_data < 0] = 0
            LRs_arr.append(_slide(torch.from_numpy(LR_data).permute(2, 0, 1).unsqueeze(0), k=slide_size).numpy() / max_val)
            HRs_arr.append(_slide(torch.from_numpy(HR_data).permute(2, 0, 1).unsqueeze(0), k=slide_size).numpy() / max_val)
            CSBS_arr.append(_slide(torch.from_numpy(CSBS_data).permute(2, 0, 1).unsqueeze(0), k=slide_size).numpy() / max_val)
        
        tp = np.float32
        self.LRs_arr = np.array(LRs_arr).astype(tp)
        self.HRs_arr = np.array(HRs_arr).astype(tp)
        self.CSBS_arr = np.array(CSBS_arr).astype(tp)
        
        print(self.LRs_arr.shape)
        print(self.HRs_arr.shape)
        print(self.CSBS_arr.shape)
    
        self.aug = RandomCrop((256, 256), p=1., cropping_mode="resample", resample="BICUBIC", keepdim=True)

    def __len__(self):
        return (self.LRs_arr.shape[0] - 1) * self.LRs_arr.shape[1]
    
    def __getitem__(self, index):
        
        len_patch = self.LRs_arr.shape[1]
        
        LR_t1 = torch.from_numpy(self.LRs_arr[index//len_patch][index%len_patch])
        HR_t1 = torch.from_numpy(self.HRs_arr[index//len_patch][index%len_patch])

        LR_t2 = torch.from_numpy(self.LRs_arr[index//len_patch + 1][index%len_patch])
        HR_t2 = torch.from_numpy(self.HRs_arr[index//len_patch + 1][index%len_patch])

        CSBS_t2 = torch.from_numpy(self.CSBS_arr[index//len_patch + 1][index%len_patch])
        
        if self.train:
            LR_t1 = self.aug(LR_t1)
            params = self.aug._params
            HR_t1 = self.aug(HR_t1, params=params)
            LR_t2 = self.aug(LR_t2, params=params)
            HR_t2 = self.aug(HR_t2, params=params)
            CSBS_t2 = self.aug(CSBS_t2, params=params)

            p_vertical_flip = 0.5 
            should_flip_vertically = torch.rand(1).item() < p_vertical_flip
            p_horizontal_flip = 0.5 
            should_flip_horizontally = torch.rand(1).item() < p_horizontal_flip
            
            if should_flip_horizontally:
                LR_t1 = F.hflip(LR_t1)
                HR_t1 = F.hflip(HR_t1)
                LR_t2 = F.hflip(LR_t2)
                HR_t2 = F.hflip(HR_t2)
                CSBS_t2 = F.hflip(CSBS_t2)


            if should_flip_vertically:
                LR_t1 = F.vflip(LR_t1)
                HR_t1 = F.vflip(HR_t1)
                LR_t2 = F.vflip(LR_t2)
                HR_t2 = F.vflip(HR_t2)
                CSBS_t2 = F.vflip(CSBS_t2)

        return {
            "LR_t1": LR_t1,
            "HR_t1": HR_t1,
            "LR_t2": LR_t2,
            "HR_t2": HR_t2,
            "CSBS": CSBS_t2
        }


class plNBUDataset(pl.LightningDataModule):
    def __init__(self, dataset_dict, batch_size, num_workers=4, pin_memory=True, ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        train_paths, val_paths = dataset_dict["train"], dataset_dict["val"]
        img_size, slide_size = dataset_dict["img_size"], dataset_dict["slide_size"]
        max_val = dataset_dict["max_value"]
        self.dataset_train = SlideDataset(train_paths, img_size, slide_size, max_val, train=True)
        self.dataset_val = SlideDataset(val_paths, img_size, slide_size, max_val, train=False)
        self.dataset_test = SlideDataset(val_paths, img_size, slide_size, max_val, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
