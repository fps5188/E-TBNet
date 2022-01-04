import glob

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from torch.utils.data import *
import cv2
from PIL import Image
from others.model_all import *
from tqdm import tqdm
from our.our_model import Model_eca
im_height = 512
im_width = 512
train_Tuberculosis = []
train_Normal = []
epochs = 100
crossNum=5
batch_size=16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

def load_data():
    train_Tuberculosis_path = glob.glob(pathname=r'../../tensorflowProject/data_set/TB_Database22/train/Tuberculosis/*.png')
    train_Normal_path = glob.glob(pathname=r'../../tensorflowProject/data_set/TB_Database22/train/Normal/*.png')
    print("-----------加载数据------------")
    for path in tqdm(train_Normal_path):
        img = cv2.imread(filename=path)
        train_Normal.append(img)
    for path in tqdm(train_Tuberculosis_path):
        img = cv2.imread(filename=path)
        train_Tuberculosis.append(img)
    print("------------训练集加载完成------------")
class Mydataset(Dataset):
    def __init__(self,x,y,trans):
        self.x = x
        self.y = torch.tensor(data=y,dtype=torch.long)
        self.trans = trans
    def __getitem__(self, index):
        img = self.x[index]
        img = Image.fromarray(obj=img)
        return self.trans(img),self.y[index]
    def __len__(self):
        return len(self.x)
#
load_data()
train_Normal = np.array(train_Normal)
train_Tuberculosis = np.array(train_Tuberculosis)
kf = StratifiedKFold(n_splits=crossNum,shuffle=True,random_state=2020)
x = np.concatenate((train_Normal,train_Tuberculosis))
y = np.array([0]*len(train_Normal)+[1]*len(train_Tuberculosis))

current_cross = 0
model_name = 'Model_eca'
print('--------------',model_name,'--------------------')
for train_index,val_index in kf.split(X=x,y=y):
    net = Model_eca()
    net.to(device=device)
    print("---------------第{}次交叉-------------------".format(current_cross+1))

    best_acc = 0.0
    current_cross+=1
    np.random.shuffle(train_index)
    np.random.shuffle(val_index)
    #训练集
    trans_train = transforms.Compose([transforms.Resize((im_height, im_width)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.RandomVerticalFlip(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = Mydataset(x[train_index],y[train_index],trans_train)
    train_num = len(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,
                              shuffle=True,num_workers=0)

    trans_val = transforms.Compose([transforms.Resize((im_height, im_width)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #验证集
    val_dataset = Mydataset(x[val_index],y[val_index],trans_val)
    val_num = len(val_dataset)
    val_loader = DataLoader(dataset=val_dataset,batch_size=batch_size,
                             shuffle=False,num_workers=0)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[20,40,60,80], gamma=0.5)

    lr_list = []
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0

        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            batch_loss = loss_function(outputs, labels.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc = torch.eq(predict_y, labels.to(device)).sum().item()
            train_acc += acc
            batch_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += batch_loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch + 1,
                                                                                epochs,
                                                                                batch_loss,
                                                                                acc / batch_size)
        #scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        # validate
        net.eval()
        val_acc = 0.0  # accumulate accurate number / epoch
        val_loss_all = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                val_loss_all += loss_function(outputs, val_labels.to(device)).item()
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = val_acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 记录损失和准确率
        train_loss_list.append(running_loss / train_steps)
        val_loss_list.append(val_loss_all / val_steps)
        train_acc_list.append(train_acc / train_num)
        val_acc_list.append(val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), './{}_{}Net.pth'.format(model_name, current_cross))

    print('Finished Training')
    print("trainloss={}".format(train_loss_list))
    print("trainaccuracy={}".format(train_acc_list))
    print("valloss={}".format(val_loss_list))
    print("valaccuracy={}".format(val_acc_list))
    print("lr={}".format(lr_list))

