import torch
from utils import check_model_forward_args
from torchmetrics.classification import Accuracy,BinaryF1Score,\
                                        AUROC, Recall, Specificity,\
                                        JaccardIndex
from torchmetrics.segmentation import DiceScore
from tqdm import tqdm
def eval_for_seg(model,val_loader,gpu_id):
    torch.cuda.set_device(gpu_id)
    with torch.no_grad():
        model.eval()
        truth_label=[]
        pred_label=[]
        for sample in tqdm(val_loader):
            image,mask,edge=sample.values()
            
            image=image.cuda()
            mask=mask.cuda()
            edge=edge.cuda()
            if check_model_forward_args(model)==2:

                pred_mask = torch.where(model(image,edge)>0.5,1,0).cpu().flatten()
            else:
                pred_mask = torch.where(model(image)>0.5,1,0).cpu().flatten()
            truth_label.extend(mask.flatten().tolist())
            pred_label.extend(pred_mask.detach().numpy().tolist())
        truth_label=torch.tensor(truth_label)
        pred_label= torch.tensor(pred_label)
        return Accuracy(task='binary')(pred_label,truth_label).item(),BinaryF1Score()(pred_label,truth_label).item(),\
            JaccardIndex(task='binary')(pred_label,truth_label).item(),Recall(task='binary')(pred_label,truth_label).item(),\
            Specificity(task='binary')(pred_label,truth_label).item(),\
            AUROC(task='binary')(pred_label,truth_label).item(),\
            DiceScore(num_classes=2)(pred_label,truth_label).item()
            