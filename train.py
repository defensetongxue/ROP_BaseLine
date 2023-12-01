import torch
from torch.utils.data import DataLoader
from util.dataset import CustomDataset
from  models import build_model
import os,json
import numpy as np
from util.metric import Metrics
from util.functions import train_epoch,val_epoch,get_optimizer,lr_sche
from configs import get_config
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
# Parse arguments
args = get_config()

os.makedirs(args.save_dir,exist_ok=True)
print("Saveing the model in {}".format(args.save_dir))
# Create the model and criterion
model= build_model(args.configs["model"])# as we are loading the exite


# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")

# early stopping
early_stop_counter = 0


# Creatr optimizer
model.train()
# Creatr optimizer
optimizer = get_optimizer(args.configs, model)
lr_scheduler=lr_sche(config=args.configs["lr_strategy"])
last_epoch = args.configs['train']['begin_epoch']

# Load the datasets
train_dataset=CustomDataset(
    split='train',data_path=args.data_path,split_name=args.split_name,resize=args.resize,norm_method=args.configs["norm_method"],enhanced=args.enhanced)
val_dataset=CustomDataset(
    split='val',data_path=args.data_path,split_name=args.split_name,resize=args.resize,norm_method=args.configs["norm_method"],
    enhanced=args.enhanced)
test_dataset=CustomDataset(
    split='test',data_path=args.data_path,split_name=args.split_name,resize=args.resize,norm_method=args.configs["norm_method"],
    enhanced=args.enhanced)
# Create the data loaders
    
train_loader = DataLoader(train_dataset, 
                          batch_size=args.configs['train']['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'],drop_last=True)
val_loader = DataLoader(val_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
test_loader=  DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])

if args.smoothing> 0.:
    from timm.loss import LabelSmoothingCrossEntropy
    criterion =LabelSmoothingCrossEntropy(args.smoothing)
    print("Using tmii official optimizier")
else:
    from models.losses import AdaptiveCrossEntropyLoss
    criterion = AdaptiveCrossEntropyLoss(train_dataset+val_dataset,device)
if args.configs['model']['name']=='inceptionv3':
    from models import incetionV3_loss
    assert args.resize>=299, "for the model inceptionv3, you should set resolusion at least 299 but now "
    
    criterion= incetionV3_loss(args.smoothing)
# init metic
metirc= Metrics(val_dataset,"Main")
print("There is {} batch size".format(args.configs["train"]['batch_size']))
print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")

early_stop_counter = 0
best_val_loss = float('inf')
best_auc=0
best_avgrecall=0
total_epoches=args.configs['train']['end_epoch']
save_model_name=args.split_name+args.configs['save_name']
saved_epoch=-1
# Training and validation loop
for epoch in range(last_epoch,total_epoches):

    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    val_loss,  metirc= val_epoch(model, val_loader, criterion, device,metirc)
    print(f"Epoch {epoch + 1}/{total_epoches}, "
      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
    #   f"Lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}"
      )
    print(metirc)
    # Early stopping
    # if metirc.average_recall >best_avgrecall:
    #     best_avgrecall= metirc.average_recall
    if metirc.auc >best_auc:
        best_auc= metirc.auc
        saved_epoch=epoch
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,save_model_name))
        print("[Save Model In Epoch {}] Model saved as {}".format(str(epoch),os.path.join(args.save_dir,save_model_name)))
    else:
        early_stop_counter += 1
        if early_stop_counter >= args.configs['train']['early_stop']:
            print("Early stopping triggered")
            break

from torch import  nn
from  util.functions import to_device
# Load the best model and evaluate
metirc=Metrics(test_dataset,"Main")
model.load_state_dict(
        torch.load(os.path.join(args.save_dir, save_model_name)))
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
loss_function=nn.CrossEntropyLoss()
model.eval()
running_loss = 0.0
all_predictions = []
all_targets = []
all_probs = []
with torch.no_grad():
    for inputs, targets, _ in val_loader:
        inputs = to_device(inputs,device)
        targets = to_device(targets,device)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        running_loss += loss.item()
        probs = torch.softmax(outputs.cpu(), dim=1).numpy()
        predictions = np.argmax(probs, axis=1)
       
        all_predictions.extend(predictions)
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs)
        
with open(os.path.join(args.data_path,'split',f'ri_{args.split_name}.json'), 'r') as f:
    split_list=json.load(f)['test_norm']
print(len(split_list))
all_predictions.extend(np.zeros((len(split_list))))
all_targets.extend(np.array([data_dict[image_name]['stage'] for image_name in split_list]))
norm_prob=np.zeros((len(split_list),4))
norm_prob[:,0]=1
all_probs.extend(norm_prob)

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
all_probs = np.vstack(all_probs)

# print(all_predictions.shape,all_probs.shape,)
metirc.update(all_predictions,all_probs,all_targets)
print(f"Best Epoch ")
print(metirc)
param={
    "model":args.configs["model"]["name"],
    "resolution": args.resize,
    "norm_method":args.norm_method,
    "smoothing":args.smoothing,
    "optimizer":args.configs["lr_strategy"],
    "weight_decay":args.configs["train"]["wd"],
    "save_epoch":saved_epoch
}
key=f"{args.configs['model']['name']}_{str(args.resize)}_{args.norm_method}_{str(args.smoothing)}_{str(args.configs['lr_strategy']['blr'])}_{str(args.configs['train']['wd'])}"
metirc._store(key,args.split_name,saved_epoch,param)