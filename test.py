import json
import os
import torch
from torch.utils.data import DataLoader
from config import get_config
from utils_ import crop_Dataset as CustomDatset
from utils_ import get_instance,plot_confusion_matrix
import models
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
def get_matrix(label, predict):
    matrix = confusion_matrix(label, predict)
    return matrix
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
name_list=[]
with torch.no_grad():
    for inputs, targets, meta in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        predicted_labels = torch.argmax(outputs, dim=1)
        
        all_targets.extend(targets.cpu().numpy())
        all_outputs.extend(predicted_labels.cpu().numpy())
        name_list.append(meta['image_name'])
acc = accuracy_score(all_targets, all_outputs)
confusion_m=confusion_matrix(all_targets, all_outputs)
print(f"Finished testing! Test acc {acc:.4f}")
print(f"all test {len(test_loader)}")
plot_confusion_matrix(confusion_m,['0','No1','1','No2','2','No3','3'],'./tmp.png')
f=open('./test_result.txt','w')
for i in range(len(all_targets)):
    f.write(f'T: {int(all_targets[i])}, P: {int(all_outputs[i])}, Image: {name_list[i]}\n')