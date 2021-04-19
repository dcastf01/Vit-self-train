import datetime
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from config import CONFIG  # ,ModelCfg
from datasets.datasets import Dataset, build_dataset, get_dataset
# from models.vt.vit import CONFIGS_VIT,VisionTransformer
# from models.vt.configs import ConfigsVit
from models.simsiam.callback import KnnMonitorInsertWandb
from models.simsiam.pl_system_simsiam import PLSystemSimSiam,
# from models.build_model import get_model
from models.simsiam_model import SimSiamModel,Classifier

# from datasets.datasets import Dataset,build_dataset

datetime.datetime.U
def main():

    wandb_logger = WandbLogger(project='vit-self-train',
                               entity='dcastf01',
                               name=datetime.datetime.utcnow().strftime("%Y-%m-%d %X"),
                            #    offline=True, #to debug
                               )
    
    # get dataloaders for training and testing
    dataset=get_dataset("C10")
    train_loader,train_classifier_loader, test_loader, num_classes = \
        build_dataset(dataset=dataset,
                      batch_size=CONFIG.BATCH_SIZE,
                      )
        
    

    # config_vit=CONFIGS_VIT[ConfigsVit.ViT_B_16]
    knn_monitor=KnnMonitorInsertWandb( train_loader,test_loader,
                                      
                                      )
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=CONFIG.PATH_CHECKPOINT,
        filename='TransformerSimsiamGeneral-{epoch:02d}-{val_loss:.4f}',
        mode="min",
        save_last=True,
        save_top_k=3,
                        )
    # backbone=VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    # config_cfg=ModelCfg()
    # model=get_model(config_cfg)
    
    system=SimSiamModel(img_size=32) 
    trainer=pl.Trainer(logger=wandb_logger,
                       gpus=-1,
                       max_epochs=CONFIG.NUM_EPOCHS,
                       precision=16,
                    #    limit_train_batches=1, #only to debug
                    #    limit_val_batches=1, #only to debug
                    #    val_check_interval=1,
                       log_gpu_memory=True,
                       callbacks=[
                    #         # early_stopping ,
                            checkpoint_callback,
                    #         # knn_monitor
                                  ]
                       )
    trainer.fit(system,train_loader)
    
    #probar la parte linear del problema
    system.eval()
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=CONFIG.PATH_CHECKPOINT,
        filename='TransformerSimsiamClassifier-{epoch:02d}-{val_loss:.2f}',
        mode="min",
        save_last=True,
        save_top_k=3,
                        )
    
    classifier = Classifier(system.model)
    trainer = pl.Trainer(logger=wandb_logger,
                       gpus=-1,
                       max_epochs=CONFIG.NUM_EPOCHS,
                       precision=16,
                    #    limit_train_batches=1, #only to debug
                    #    limit_val_batches=1, #only to debug
                    #    val_check_interval=1,
                       log_gpu_memory=True,
                       callbacks=[
                            # early_stopping ,
                            checkpoint_callback,
                            # knn_monitor
                                  ]
                            )
    
    
    trainer.fit(
        classifier,
        train_classifier_loader,
        test_loader
                    )
    
        
        
        
        
if __name__ == "__main__":
    main()
