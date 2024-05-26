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
args.configs['model']['num_classes']=3
# select the lr and wd for hyper parameter adjustment
args.configs["lr_strategy"]["lr"]=args.lr
args.configs['train']['lr']=args.lr
args.configs['train']['wd']=args.wd
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
    split='train',data_path=args.data_path,split_name='stage_1',resize=args.resize,norm_method=args.configs["norm_method"])
test_dataset=CustomDataset(
    split='test',data_path=args.data_path,split_name='stage_1',resize=args.resize,norm_method=args.configs["norm_method"])
# Create the data loaders
    
train_loader = DataLoader(train_dataset, 
                          batch_size=args.configs['train']['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'],drop_last=True)
test_loader=  DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])

if args.smoothing> 0.:
    from timm.loss import LabelSmoothingCrossEntropy
    criterion =LabelSmoothingCrossEntropy(args.smoothing)
    print("Using tmii official optimizier")
else:
    raise
    from models.losses import AdaptiveCrossEntropyLoss
    criterion = AdaptiveCrossEntropyLoss(train_dataset+val_dataset,device)
if args.configs['model']['name']=='inceptionv3':
    from models import incetionV3_loss
    assert args.resize>=299, "for the model inceptionv3, you should set resolusion at least 299 but now "
    
    criterion= incetionV3_loss(args.smoothing)
# init metic
metirc= Metrics("Main",num_class=3)
print("There is {} batch size".format(args.configs["train"]['batch_size']))
early_stop_counter = 0
best_val_loss = float('inf')
best_auc=0
best_avgrecall=0
total_epoches=args.configs['train']['end_epoch']
save_model_name=args.split_name+args.configs['save_name']
saved_epoch=-1
# Training and validation loop
record={}
for epoch in range(last_epoch,total_epoches):
    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    val_loss, metirc=val_epoch(model, test_loader, criterion, device,metirc)
    record[epoch]={
        'acc':f"{metirc.accuracy :.4f}",
        'auc':f"{metirc.auc :.4f}"}
    print(f"{metirc.auc :.4f}")
with open('./experiments/0508.json','w') as f:
    json.dump(record,f)