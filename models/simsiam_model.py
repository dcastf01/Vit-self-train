import pytorch_lightning as pl
import torch
import torch.nn as nn
import lightly

# from models.vit import VisionTransformerGenerator

from models.vt.lucidrains import VisionTransformerGenerator
class SimSiamModel(pl.LightningModule):
    def __init__(self,img_size):
        super().__init__()
        
        #create a vit backbone and remove the classification head
        
        vit=VisionTransformerGenerator( img_size=img_size )
        backbone=nn.Sequential(*list(vit.children())[:-1],
                            #    nn.AdaptiveAvgPool2d(1),
                               )
        # resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=8)
        # backbone = nn.Sequential(
        #     *list(resnet.children())[:-1],
        #     nn.AdaptiveAvgPool2d(1),
        #         )
        #create a 
        self.model=lightly.models.SimSiam(
                            backbone,
                            num_ftrs=768
                            
                            )
        
        self.criterion=lightly.loss.SymNegCosineSimilarityLoss()
        
    def forward(self,x):
        return self.model(x)
    
    def training_step(self,batch,batch_idx):
        (x0,x1),_,_=batch
        y0,y1=self.model(x0,x1)
        loss=self.criterion(y0,y1)
        
        self.log("train_loss",loss,on_step=False,on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        # scale the learning rate
        lr = 0.05 * 32 / 256
        # use SGD with momentum and weight decay
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4
                    )
        return optimizer
    
class Classifier(pl.LightningModule):     
    
        

    def __init__(self, model):
        super().__init__()
        # create a moco based on ResNet
        self.model = model

        # freeze the layers of moco
        for p in self.model.parameters():  # reset requires_grad
            p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(512, 10) #10 es el numero de clases

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.model.backbone(x).squeeze()
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=30.)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]