
import os
from datetime import datetime
import torch.distributed as dist
import yaml
from pytorch_lightning import loggers

import os
from datetime import datetime
import torch.distributed as dist
from pytorch_lightning import loggers

def get_logger_and_paths(ckpt_path, fold, timestamp=None):
    if timestamp is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else: date_str = date_str = timestamp
    
    base_dir = os.path.join(ckpt_path, f"fold_{fold}_{date_str}")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    if dist.is_initialized(): dist.barrier()

    tb_logger = loggers.TensorBoardLogger(save_dir=base_dir, name="")
    csv_logger = loggers.CSVLogger(save_dir=base_dir, name="")

    return base_dir, [tb_logger, csv_logger]

def save_hparams(base_dir, args):
    if not dist.is_initialized() or dist.get_rank() == 0:
        hparam_path = os.path.join(base_dir, "hparams.yaml")
        with open(hparam_path, "w") as f:
            yaml.dump(vars(args), f)