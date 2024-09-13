from datetime import datetime
import os


def get_pairs(data_path, num_train, key_Landsat, key_MODIS, key_CSBS):
    get_sorted_files = lambda x,k: sorted(os.listdir(x), key=k)
    LR_sorted_files = get_sorted_files(os.path.join(data_path, "MODIS"), key_MODIS)
    HR_sorted_files = get_sorted_files(os.path.join(data_path, "Landsat"), key_Landsat)
    CSBS_sorted_files = get_sorted_files(os.path.join(data_path, "CSBS"), key_CSBS)
    LRs = []
    HRs = []
    CSBS = []
    for LR_name, HR_name, CSBS_name in zip(LR_sorted_files, HR_sorted_files, CSBS_sorted_files):
        LRs.append(os.path.abspath(os.path.join(data_path, "MODIS", LR_name)))
        HRs.append(os.path.abspath(os.path.join(data_path, "Landsat", HR_name)))
        CSBS.append(os.path.abspath(os.path.join(data_path, "CSBS", CSBS_name)))

    train_val_dict =  {
        "train": {
            "LRs": LRs[:num_train],
            "HRs": HRs[:num_train],
            "CSBS": CSBS[:num_train]
        },
        "val":{
            "LRs": LRs[num_train:],
            "HRs": HRs[num_train:],
            "CSBS": CSBS[num_train:]
        }
    }

    return train_val_dict



CIA_data = get_pairs(data_path="data/CIA", 
                    num_train=11, 
                    key_Landsat=lambda x: x[13:21],
                    key_MODIS=lambda x: x[9:16],
                    key_CSBS=lambda x: x[9:16],
                    )

LGC_data = get_pairs(data_path="data/LGC", 
                    num_train=9, 
                    key_Landsat=lambda x: x[:8],
                    key_MODIS=lambda x: x[9:16],
                    key_CSBS=lambda x: x[:8],
                    )

AHB_data = get_pairs(data_path="data/Datasets/AHB", 
                    num_train=20, 
                    key_Landsat=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    key_MODIS=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    key_CSBS=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    )

Daxing_data = get_pairs(data_path="data/Datasets/Daxing", 
                    num_train=20, 
                    key_Landsat=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    key_MODIS=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    key_CSBS=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    )

Tianjin_data = get_pairs(data_path="data/Datasets/Tianjin", 
                    num_train=20, 
                    key_Landsat=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    key_MODIS=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    key_CSBS=lambda f: datetime.strptime(f[2:-4], '%Y-%m-%d'),
                    )

name2data = {
    "CIA": {
        "train": CIA_data["train"],
        "val": CIA_data["val"],
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (1792, 1280),
        "slide_size": 512,
        "max_value": 10000.,
    },
    "LGC": {
        "train": LGC_data["train"],
        "val": LGC_data["val"],
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (2560, 3072),
        "slide_size": 512,
        "max_value": 10000.,
    },
    "AHB": {
        "train": AHB_data["train"],
        "val": AHB_data["val"],
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (2480, 2800),
        "slide_size": 512,
        "max_value": 255.,
    },
    "DX": {
        "train": Daxing_data["train"],
        "val": Daxing_data["val"],
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (1640, 1640),
        "slide_size": 512,
        "max_value": 255.,
    },
    "TJ": {
        "train": Tianjin_data["train"],
        "val": Tianjin_data["val"],
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (2100, 1970),
        "slide_size": 512,
        "max_value": 255.,
    }
}
