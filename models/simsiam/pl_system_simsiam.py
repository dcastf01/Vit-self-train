
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from config import CONFIG

from models.simsiam.knn_monitor import knn_monitor
class PLSystemSimSiam(pl.LightningModule):
    def __init__(self,
                 model,
                #  metrics_collection:torchmetrics.MetricCollection,
                #   loss_fn=nn.CrossEntropyLoss(),
                  lr=CONFIG.LEARNING_RATE):
        
        super().__init__()
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        self.model=model

        # self.loss_fn=loss_fn
        # self.train_metrics=metrics_collection.clone()
        # self.valid_metrics=metrics_collection.clone()
        
        # log hyperparameters
        self.save_hyperparameters()
        
        self.lr=lr
            
    def forward(self,x):
        
        x=self.model(x)
        
        return x #quizá existe el problema de que la salida es un diccionario
    
    def training_step(self,batch,batch_idx):
        #llegan dos imagenes en train
        (x1,x2),targets=batch
        
        data_dict=self.model(x1,x2)
        loss=data_dict["loss"]
        self.log("train_loss",loss,on_step=False,on_epoch=True)
        # self.logger.experiment.add_scalar("train_loss",loss,on_step=False,on_epoch=True)
      
        #lo ideal sería hacer algo tipo loss,preds=self.model(x) de esta manera al modelo se le 
        #incluiria el loss fn y así podremos hacer la tripleta
        # loss=self.loss_fn(preds,targets)
        
        # metric_value=self.train_metrics(preds.softmax(dim=-1),targets)
        # self.insert_each_metric_value_into_dict(data_dict,prefix="train")
        # self.log('train_loss',loss)
        # self.log('train_metrics',metric_value)
        
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        pass
        # #test esta dando mal porque solo devuelve una imagen y su etiqueta
        # x, targets = batch
        
        # data_dict=self.model(x,x) #arreglo feo y temporal
        # loss=data_dict["loss"]
        # # preds=data_dict["preds"]
        
        # if isinstance(targets,list):
        #     targets=targets[0]

        # metric_value=self.valid_metrics(preds.softmax(dim=-1),targets)
        # data_dict.pop("preds")
        # data_dict={**data_dict,**metric_value}
        # #########CREAR UNA FUNCIÓN QUE COJA METRICA Y DATA DICT, Y SUELTE LA PREDICCION 
        # # ##################Y GENERE EL LOG CORRESPONDIENTE
        # # Log validation loss (will be automatically averaged over an epoch)
        # self.insert_each_metric_value_into_dict(data_dict,prefix="valid")
       

        # Log metrics
        #self.log('valid_acc', self.accuracy(logits, y))
   
        
        
            
    def configure_optimizers(self):
        
        optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    
    def insert_each_metric_value_into_dict(self,data_dict:dict,prefix=None):
        
        for metric,value in data_dict.items():
            self.log("_".join([prefix,metric]),value)