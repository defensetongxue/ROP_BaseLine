import torch
from torch.utils.data import DataLoader
from util.dataset import CustomDataset
from  models import build_model
import os,json
import numpy as np
from util.metric import Metrics
from util.functions import to_device
from configs import get_config
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
# Parse arguments
args = get_config()
args.data_path='../autodl-tmp/ROP_shen'
args.configs['model']['num_classes']=2
os.makedirs(args.save_dir,exist_ok=True)
print("Saveing the model in {}".format(args.save_dir))
# Create the model and criterion
model= build_model(args.configs["model"])# as we are loading the exite

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")
save_model_name='all'+args.configs['save_name']
model.eval()
# Creatr optimizer
test_dataset=CustomDataset(
    split='test',data_path=args.data_path,split_name='clr',resize=args.resize,norm_method=args.configs["norm_method"],
    enhanced=args.enhanced,bin=True)
# Create the data loaders
    
test_loader=  DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])


# Load the best model and evaluate
metirc=Metrics("Main",num_class=2)
model.load_state_dict(
        torch.load(os.path.join(args.save_dir, save_model_name)))
model.eval()
running_loss = 0.0
all_predictions = []
all_targets = []
all_probs = []
with torch.no_grad():
    for inputs, targets, _ in test_loader:
        inputs = to_device(inputs,device)
        targets = to_device(targets,device)
        outputs = model(inputs)
        probs = torch.softmax(outputs.cpu(), dim=1).numpy()
        predictions = np.argmax(probs, axis=1)
       
        all_predictions.extend(predictions)
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs)
        
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
all_probs = np.vstack(all_probs)
# print(all_predictions.shape,all_probs.shape,)
metirc.update(all_predictions,all_probs,all_targets)
print(metirc)
param={
    "model":args.configs["model"]["name"],
    "resolution": args.resize,
    "norm_method":args.norm_method,
    "smoothing":args.smoothing,
    "optimizer":args.configs["lr_strategy"],
    "weight_decay":args.configs["train"]["wd"]
}
key=f"{args.configs['model']['name']}_{str(args.resize)}_{args.norm_method}_{str(args.smoothing)}_{str(args.configs['lr_strategy']['lr'])}_{str(args.configs['train']['wd'])}"
metirc._store(key,args.split_name,param,save_path='./experiments/sz_ridge.json')