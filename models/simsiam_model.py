import pytorch_lightning as pl
import torch
import torch.nn as nn
import lightly
import torchvision
# from models.vit import VisionTransformerGenerator
from metric import CollapseLevel
from torchmetrics import MetricCollection
from models.vt.lucidrains import VisionTransformerGenerator
from models.benchmark_module import BenchmarkModule
from config import CONFIG

class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN,img_size=32):
        super().__init__(dataloader_kNN)
        # create a ResNet backbone and remove the classification head
        vit=VisionTransformerGenerator( img_size=img_size )
        self.backbone=nn.Sequential(*list(vit.children())[:-1],
                            #    nn.AdaptiveAvgPool2d(1),
                               )
        # resnet = lightly.models.ResNetGenerator('resnet-18')
        # self.backbone = nn.Sequential(
        #     *list(resnet.children())[:-1],
        #     nn.AdaptiveAvgPool2d(1),
        # )
        # create a simsiam model based on ResNet
        self.model = \
            lightly.models.SimSiam(self.backbone, num_ftrs=768, num_mlp_layers=2)
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        self.metric=MetricCollection({"CollapseLevel":CollapseLevel()})
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.model(x0, x1)
        
        loss = self.criterion(y0, y1)
        
        self.log('train_loss_ssl', loss)
        self.metric(y0[0]) #solo proyectamos la proyeccion, no la prediccion

      
        self.log("Collapse Level",self.metric["CollapseLevel"],on_step=True,on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, CONFIG.NUM_EPOCHS)
        return [optim], [scheduler]
    

    
class Classifier(pl.LightningModule):     
    
        

    def __init__(self, model,max_epochs):
        super().__init__()
        # create a moco based on ResNet
        self.model = model
        self.max_epochs=max_epochs
        # freeze the layers of moco
        for p in self.model.parameters():  # reset requires_grad
            p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(768, 10) #10 es el numero de clases

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.model.backbone(x).squeeze()
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    # def custom_histogram_weights(self):
    #     for name, params in self.named_parameters():
    #         self.logger.experiment.add_histogram(
    #             name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        self.accuracy(y_hat.softmax(dim=1), y)
        self.log('train_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)
        
        return loss
    
    # def training_epoch_end(self, outputs):
    #     self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('valid_loss_fc', loss)
        self.accuracy(y_hat.softmax(dim=1), y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=60.)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]