import os
import json

import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from others.model_all import *
from our.our_model import Model_eca
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


    image_path = r"G:\dataset\ChinaSet_AllFiles"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    model_name = "Model_eca"
    net = Model_eca()
    print(net)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.03)
    scheduler = MultiStepLR(optimizer=optimizer,milestones=[30,50,70,90,110,140],gamma=0.5)#0015 0.0005 #0.00025,0.0001,0.00005

    epochs = 100
    best_acc = 0.0
    lr_list=[]
    train_loss_list=[]
    val_loss_list=[]
    train_acc_list=[]
    val_acc_list=[]
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
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
            acc= torch.eq(predict_y, labels.to(device)).sum().item()
            train_acc+=acc
            batch_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += batch_loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     batch_loss,
                                                                     acc/batch_size)
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        # validate
        net.eval()
        val_acc = 0.0  # accumulate accurate number / epoch
        val_loss_all=0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                val_loss_all+= loss_function(outputs, val_labels.to(device)).item()
                val_acc +=torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = val_acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        #记录损失和准确率
        train_loss_list.append(running_loss/train_steps)
        val_loss_list.append(val_loss_all/val_steps)
        train_acc_list.append(train_acc/train_num)
        val_acc_list.append(val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            save_path = './{}Net_{}_.pth'.format(model_name,val_accurate)
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    print("trainloss={}".format(train_loss_list))
    print("trainaccuracy={}".format(train_acc_list))
    print("valloss={}".format(val_loss_list))
    print("valaccuracy={}".format(val_acc_list))
    print("lr={}".format(lr_list))

if __name__ == '__main__':
    main()