import json
import os
import torch
from torch.utils.data import DataLoader
from config import get_config
from utils_ import crop_Dataset as CustomDatset
from utils_ import get_instance
import models
from sklearn.metrics import roc_auc_score, accuracy_score

# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.configs.MODEL.NAME,args.configs,
                         num_classes=args.configs.NUM_CLASS)
criterion=torch.nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(args.save_name))
print(f"load the checkpoint in {args.save_name}")
model.eval()

# Create the dataset and data loader
data_path=os.path.join(args.path_tar)
test_dataset = CustomDatset(data_path,split='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# Create the visualizations directory if it doesn't exist
all_targets = []
all_outputs = []

with torch.no_grad():
    for inputs, targets, meta in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        predicted_labels = torch.argmax(outputs, dim=1)
        
        all_targets.extend(targets.cpu().numpy())
        all_outputs.extend(predicted_labels.cpu().numpy())
acc = accuracy_score(all_targets, all_outputs)
def sensitive_score(label,predict):
    success_cnt=0
    ill_cnt=0
    for i,j in zip(label,predict):
        i,j=int(i),int(j)
        if i>0:
            ill_cnt+=1
            if j>0:
                success_cnt+=1 
    return success_cnt/ill_cnt

print(f"Finished testing! Test acc {acc:.4f}")
# all_targets=all_targets.squeeze()
# all_outputs=all_outputs.squeeze()
print(f"all test {len(test_loader)} sens {(sensitive_score(all_targets,all_outputs)):.4f}")
