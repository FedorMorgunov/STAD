# eval.py
import argparse
import os


import pytorch_lightning as pl
import yaml


from models.my_transformer_ad import TransformerAD


from utils.argparser import init_args
from utils.dataset import get_dataset_and_loader


if __name__== '__main__':
  
   parser = argparse.ArgumentParser(description='MoCoDAD')
   parser.add_argument('-c', '--config', type=str, required=True)
   args_ = parser.parse_args()


   args = yaml.load(open(args_.config), Loader=yaml.FullLoader)
   args = argparse.Namespace(**args)
   args = init_args(args)


   # Create the new model
   model = TransformerAD(args)
  
   if args.load_tensors:
       # If you have a feature to skip direct inference and load saved predictions
       model.test_on_saved_tensors(split_name=args.split)
   else:
       # Load test data
       print('Loading data and creating loaders.....')
       ckpt_path = os.path.join(args.ckpt_dir, args.load_ckpt)
       dataset, loader, _, _ = get_dataset_and_loader(args, split=args.split)
      
       trainer = pl.Trainer(
           accelerator=args.accelerator,
           devices=args.devices[:1],
           default_root_dir=args.ckpt_dir,
           max_epochs=1,
           logger=False
       )
       out = trainer.test(model, dataloaders=loader, ckpt_path=ckpt_path)
