
import argparse
import pytorch_lightning as pl
import os
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from name2data import name2data
from SlideDataset import plNBUDataset
from SDCS import SDCSModel
from util import check_and_make


def get_args_parser():
    parser = argparse.ArgumentParser('SDCS training', add_help=False)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--dataset', default="CIA", choices=["CIA", "LGC", "AHB", "DX", "TJ"], type=str)
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    model_name = "STF." + SDCSModel.__name__ 

    output_dir = f"log_m={model_name}_sd={args.seed}_d={args.dataset}"

    check_and_make(output_dir)
    seed_everything(args.seed)

    dataset = plNBUDataset(name2data[args.dataset],
                           args.batch_size,
                           args.num_workers,
                           args.pin_mem
                           )
    model = SDCSModel(lr=args.lr,
                      epochs=args.epochs,
                      bands=name2data[args.dataset]["band"],
                      rgb_c=name2data[args.dataset]["rgb_c"],
                      dataname=args.dataset
                    )
    
    if args.wandb:
        wandb_logger = WandbLogger(project=model_name, name=output_dir, save_dir=output_dir)
    else:
        wandb_logger = [CSVLogger(name=output_dir, save_dir=output_dir)]
        wandb_logger.append(TensorBoardLogger(name=output_dir, save_dir=output_dir))
                
    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       monitor='val/SSIM_mean',
                                       mode="max",
                                       save_top_k=1,
                                       auto_insert_metric_name=False,
                                       filename='ep={epoch}_SSIM={val/SSIM_mean:.4f}',
                                       save_last=True,
                                       every_n_epochs=args.test_freq
                                       )

    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator="gpu",
                         devices=[args.device],
                         logger=wandb_logger,
                         check_val_every_n_epoch=args.test_freq,
                         callbacks=[model_checkpoint],
                         )

    if not args.test:
        trainer.fit(model, dataset)
        trainer.test(ckpt_path="best", datamodule=dataset)
    else:
        trainer.test(model, ckpt_path=args.ckpt, datamodule=dataset)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
