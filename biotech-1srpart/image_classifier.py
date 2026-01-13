''' 
Image Classifier with Transfer Learning and CNN
''' 
!pip install -U torch torchvision torchtext torchaudio --quiet
!pip install -U pytorch-lightning --quiet
!pip install opendatasets --upgrade --quiet

import os,json, logging, zipfile, shutil
from pathlib import Path
import opendatasets as od, pandas as pd,  numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
pd.__version__, np.__version__, torch.__version__, pl.__version__

from google.colab import files
files.upload()

"""Data processing"""

DATA_PATH = Path('data')
KAGGLE_JSON = DATA_PATH/'kaggle.json'
IS_KAGGLE_KEY = KAGGLE_JSON.exists()
KAGGLE_API = None
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c histopathologic-cancer-detection
!unzip histopathologic-cancer-detection.zip

cancer_labels = pd.read_csv('train_labels.csv')

''' 0 for normal cases, 1 for abnormal '''
cancer_labels.head()[:2], cancer_labels.label.value_counts()

''' imgs in training/testing datasets '''
len(os.listdir('train')), len(os.listdir('test'))

np.random.seed(0)
train_imgs_lst = os.listdir('train')
selected_img_lst = []
for img in np.random.choice(train_imgs_lst, 10000):
  selected_img_lst.append(img)
fig = plt.figure(figsize=(15, 5))
for index, img in enumerate(np.random.choice(selected_img_lst, 8)):
  ax = fig.add_subplot(1, 8, index+1, xticks=[], yticks=[])
  img_n = Image.open('train/' +img)
  plt.imshow(img_n)
  label = cancer_labels.loc[cancer_labels['id'] == img.split('.')[0],'label'].values[0]
  ax.set_title("label: {label}")

np.random.seed(0)
np.random.shuffle(selected_img_lst)
cancer_train, cancer_test = selected_img_lst[:9000], selected_img_lst[9000:]
for filename in cancer_train:
  src = os.path.join('train', filename)
  dst = os.path.join('train_ds', filename)
  shutil.copyfile(src, dst)
for filename in cancer_test:
  src = os.path.join('train', filename)
  dst = os.path.join('test_ds', filename)
  shutil.copyfile(src, dst)

data_transform_train = T.Compose([
  T.CenterCrop(32), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ToTensor()])
data_transform_test = T.Compose([
  T.CenterCrop(32), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ToTensor()])

id_list, label_list = [], []
selected_img_labels=pd.DataFrame()

for img in selected_img_lst:
  label_tuple = cancer_labels.loc[cancer_labels['id'] == img.split('.')[0]]
  id_list.append(label_tuple['id'].values[0])
  label_list.append(label_tuple['label'].values[0])

selected_img_labels['id'] = id_list
selected_img_labels['label'] = label_list
img_label_dict = {k:v for k, v in zip(selected_img_labels.id, selected_img_labels.label)}
selected_img_labels.head()[:2], #img_label_dict

class CancerDataset(Dataset):
  def __init__(self, data_folder,
               transform = T.Compose([T.CenterCrop(32),T.ToTensor()]), dict_labels={}):
    self.data_folder = data_folder
    self.list_image_files = [s for s in os.listdir(data_folder)]
    self.transform = transform
    self.dict_labels = dict_labels
    self.labels = [dict_labels[i.split('.')[0]] for i in self.list_image_files]

  def __len__(self):
    return len(self.list_image_files)

  def __getitem__(self, idx):
    img_name = os.path.join(self.data_folder, self.list_image_files[idx])
    image = Image.open(img_name)
    image = self.transform(image)
    img_name_short = self.list_image_files[idx].split('.')[0]

    label = self.dict_labels[img_name_short]
    return image, label

train_set = CancerDataset(
    data_folder='/content/train_ds/',
    transform=data_transform_train, dict_labels=img_label_dict)
test_set = CancerDataset(
    data_folder='/content/test_ds/',
    transform=data_transform_train, dict_labels=img_label_dict)

batch_size = 256

train_dataloader = DataLoader(
    train_set, batch_size, num_workers=2, pin_memory=True, shuffle=True)
test_dataloader = DataLoader(
    test_set, batch_size, num_workers=2, pin_memory=True)

"""Transfer Learning"""

from torchvision.models import resnet50
resnet50(pretrained=True)

class TLImageClassifier(pl.LightningModule):
  def __init__(self, learning_rate = 0.001):
    super().__init__()

    self.learning_rate = learning_rate
    self.loss = nn.CrossEntropyLoss()
    self.pretrain_model = resnet50(pretrained=True)
    self.pretrain_model.eval()
    for param in self.pretrain_model.parameters():
      param.requires_grad = False

    self.pretrain_model.fc = nn.Linear(2048, 2)

  def forward(self, input):
    output=self.pretrain_model(input)
    return output

  def training_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    preds = torch.argmax(outputs, dim=1)
    loss = self.loss(outputs, targets)
    self.log('train_loss', loss)
    return {"loss": loss}


  def test_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self.forward(inputs)
    loss = self.loss(outputs, targets)
    return {"test_loss": loss}

  def configure_optimizers(self):
    params = self.parameters()
    optimizer = optim.Adam(params=params, lr = self.learning_rate)
    return optimizer

!mkdir transfer_learning
!chmod 775 transfer_learning

model = TLImageClassifier()
trainer = pl.Trainer(fast_dev_run=True, devices=1)
trainer.fit(model, train_dataloader)
checkpoint_dir = "/content/transfer_learning"
checkpoint_callback =  pl.callbacks.ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="epoch={epoch}-val_loss={val_loss:.2f}",  # Filename format
    monitor="val_loss",
    verbose=True,
    save_last=True,
    save_top_k=3,
    mode="min",
)

model = TLImageClassifier()
trainer = pl.Trainer(
    default_root_dir=checkpoint_dir, devices=1,
    callbacks=[checkpoint_callback],
    log_every_n_steps=30,
    max_epochs=10)

trainer.fit(model, train_dataloaders=train_dataloader)

trainer.test(dataloaders=test_dataloader, ckpt_path='/content/transfer_learning/last.ckpt')

model.eval()
preds = []
for batch_i, (data, target) in enumerate(test_dataloader):
  target = target.cpu()
  #target = target.cuda()
  output = model(data)
  pr = output[:,1].detach().cpu().numpy()
  for i in pr:
    preds.append(i)

test_preds = pd.DataFrame({'imgs': test_set, 'labels': test_set.labels,  'preds': preds})

test_preds

#test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])

test_preds['predictions'] = 1
test_preds.loc[test_preds['preds'] < 0, 'predictions'] = 0
test_preds.head()

len(np.where(test_preds['labels'] == test_preds['predictions'])[0])/test_preds.shape[0]

"""Image Classifier with CNN"""

class CNNImageClassifier(pl.LightningModule):
  def __init__(self, learning_rate = 0.001):
    super().__init__()

    self.learning_rate = learning_rate

    self.conv_layer1 = nn.Conv2d(
        in_channels = 3, out_channels=3, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2)
    self.conv_layer2 = nn.Conv2d(
        in_channels=3, out_channels=6, kernel_size=3, stride=1,padding=1)
    self.relu2 = nn.ReLU()
    self.fully_connected_1 = nn.Linear(in_features=16 * 16 * 6,out_features=1000)
    self.fully_connected_2 = nn.Linear(in_features=1000, out_features=500)
    self.fully_connected_3 = nn.Linear(in_features=500, out_features=250)
    self.fully_connected_4 = nn.Linear(in_features=250, out_features=120)
    self.fully_connected_5 = nn.Linear(in_features=120, out_features=60)
    self.fully_connected_6 = nn.Linear(in_features=60, out_features=2)
    self.loss = nn.CrossEntropyLoss()
    self.save_hyperparameters()

  def forward(self, input):
    output = self.conv_layer1(input)
    output = self.relu1(output)
    output = self.pool(output)
    output = self.conv_layer2(output)
    output = self.relu2(output)
    output = output.view(-1, 6*16*16)
    output = self.fully_connected_1(output)
    output = self.fully_connected_2(output)
    output = self.fully_connected_3(output)
    output = self.fully_connected_4(output)
    output = self.fully_connected_5(output)
    output = self.fully_connected_6(output)
    return output

  def training_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    #train_accuracy = accuracy(outputs, targets)
    loss = self.loss(outputs, targets)
    #self.log('train_accuracy', train_accuracy, prog_bar=True)
    self.log('train_loss', loss)
    #return {"loss": loss, "train_accuracy": train_accuracy}
    return {"loss": loss}

  def test_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self.forward(inputs)
    #test_accuracy = accuracy(outputs, targets)
    loss = self.loss(outputs, targets)
    #self.log('test_accuracy', test_accuracy)
    #return {"test_loss": loss, "test_accuracy": test_accuracy}
    return {"test_loss": loss}

  def configure_optimizers(self):
    params = self.parameters()
    optimizer = optim.Adam(params=params, lr = self.learning_rate)
    return optimizer

    # Calculate accuracy for each batch at a time
  def binary_accuracy(self, outputs, targets):
    _, outputs = torch.max(outputs, 1)
    correct_results_sum = (outputs == targets).sum().float()
    accuracy = correct_results_sum/targets.shape[0]
    return accuracy

  def predict_step(self, batch, batch_idx ):
    return self(batch)

device = 'cpu'
!mkdir cnn
!chmod 775 cnn

next(iter(train_dataloader))

model = CNNImageClassifier()
trainer = pl.Trainer(fast_dev_run=True, devices=1)
trainer.fit(model, train_dataloaders=train_dataloader)

checkpoint_dir = "/content/cnn"
checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=10)

model = CNNImageClassifier()
trainer = pl.Trainer(
    default_root_dir=checkpoint_dir,
    devices=1,
    callbacks=[checkpoint_callback],
    log_every_n_steps=25,
    max_epochs=500)

model.eval()
preds = []
for batch_i, (data, target) in enumerate(test_dataloader):
  #data, target = data.cuda(), target.cuda()
  #output = model.cuda()(data)
  data, target = data.cpu(), target.cpu()
  output = model.cpu()(data)

  pr = output[:,1].detach().cpu().numpy()
  for i in pr:
    preds.append(i)

test_preds = pd.DataFrame(
  {'imgs': test_set.list_image_files, 'labels':test_set.labels, 'preds': preds})
test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])
test_preds.head()

test_preds['predictions'] = 1
test_preds.loc[test_preds['preds'] < 0, 'predictions'] = 0
test_preds.shape

test_preds.head()
len(np.where(test_preds['labels'] == test_preds['predictions'])[0])/test_preds.shape[0]