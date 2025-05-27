import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from utils import check_model_forward_args
from tqdm import tqdm
from eval import eval_for_seg
import logging
from datetime import datetime
class Trainer:
    def __init__(self,model,train_loader
                 ,val_loader,criterion,optimizer,scheduler,gpu_id,name,save_dir='./checkpoints'):
        self.model=model
        self.train_loader=train_loader
        self.val_loader= val_loader
        self.criterion=criterion
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.gpu_id=gpu_id
        self.save_dir=save_dir
        self.name=name

        model_class_name = type(self.model).__name__
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"exp_on_{self.name}_{model_class_name}_{timestamp}.log"
        log_path = os.path.join('./logs', log_filename)
        os.makedirs('./logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger()
    def train(self,epochs=100):
        torch.cuda.set_device(self.gpu_id)
        self.model.cuda()
        self.model.train()
        for e in range(epochs):
            training_loss=0
            for sample in tqdm(self.train_loader):
                image,mask,edge=sample.values()
                image = image.cuda()
                mask = mask.cuda()
                edge = edge.cuda()

                if check_model_forward_args(self.model)==2:
                    pred_mask = self.model(image,edge)
                else:
                    pred_mask = self.model(image)
                loss = self.criterion(pred_mask,mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                training_loss+=loss.item()
            self.scheduler.step(training_loss)
            acc,f1,iou,recall,spe,auc,dice=eval_for_seg(self.model,self.val_loader,self.gpu_id)
            print(f'Epoch [{e}/{epochs}]: {self.name} has training loss{training_loss},test acc',
                    acc,f'test_f1 {f1}',f'test_iou {iou}', f'test_sen {recall}',
                    f'test_spe {spe}',f'test_auc {auc}', f'test_dice {dice}',
                    )
            
