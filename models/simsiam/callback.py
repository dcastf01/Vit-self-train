from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F 
import torch
from models.simsiam.knn_monitor import knn_monitor
import wandb
from tqdm import tqdm
# from copy import copy
import copy
from typing import Optional
class KnnMonitorInsertWandb(Callback):
    
    def __init__(self,
                 train_data_loader,
                 test_data_loader,
                 ):
        super().__init__()
        self.memory_data_loader=copy.deepcopy(train_data_loader)
        self.test_data_loader=test_data_loader
        self.memory_data_loader.dataset.transform=self.test_data_loader.dataset.transform
        every_n_train_steps=2
        every_n_val_epochs=2
        self.__init_triggers(every_n_train_steps, every_n_val_epochs)
   
    def on_validation_epoch_end(self,trainer, pl_module):
        net=pl_module.model.backbone
        memory_data_loader=self.memory_data_loader
        test_data_loader=self.test_data_loader
        epoch=pl_module.current_epoch
        accuracy=knn_monitor(net, memory_data_loader, test_data_loader, epoch, k=200, t=0.1, hide_progress=False)
        pl_module.log("accuracy_knn_monitor",accuracy)
        pl_module.log("epoch",epoch)
                 
                  
        # wandb.log({"accuracy_knn_monitor":accuracy,
        #            "epoch":epoch},
        #           )
        
    def __init_triggers(
        self, every_n_train_steps: Optional[int], every_n_val_epochs: Optional[int], 
    ) -> None:

        # Default to running once after each validation epoch if neither
        # every_n_train_steps nor every_n_val_epochs is set
        if every_n_train_steps is None and every_n_val_epochs is None:
            self._every_n_val_epochs = 1
            self._every_n_train_steps = 0
            log.debug("Both every_n_train_steps and every_n_val_epochs are not set. Setting every_n_val_epochs=1")
        else:
            self._every_n_val_epochs = every_n_val_epochs or 0
            self._every_n_train_steps = every_n_train_steps or 0

        # period takes precedence over every_n_val_epochs for backwards compatibility
      

        self._period = self._every_n_val_epochs