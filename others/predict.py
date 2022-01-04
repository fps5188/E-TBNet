import pickle
import time

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from others.model_all import *
from tool_myself import genConfusionMatrix

batch_size=8
#You can change the path to the weight path of the model you want to execute.
weights_path = r"weight/MobileNetV2/mobilenet_v2_1Net.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output = open(r'test_dataset.pkl', 'rb')
test_dataset = pickle.load(output)
output.close()

testdata_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0)

test_num = len(test_dataset)
# You can change to the model you want to evaluate.

model = get_mobilenet_v2().to(device)

model.load_state_dict(torch.load(weights_path, map_location=device))
acc = 0.0
pre = np.array([], dtype=int)
tar = np.array([], dtype=int)
roc_pre = np.array([], dtype=int)
model.eval()
times = []
with torch.no_grad():
    val_bar = tqdm(testdata_loader)
    for val_data in val_bar:
        val_images, val_labels = val_data
        start = time.time()
        outputs = model(val_images.to(device))
        end = time.time()
        times.append((end - start) * 1000)
        #print("每个batch一张图耗时(ms):", (end - start) * 1000 / batch_size)
        predict_y = torch.max(outputs, dim=1)[1]
        pre = np.append(pre, predict_y.cpu().numpy())
        tar = np.append(tar, val_labels.cpu().numpy())
        temp = outputs.cpu().numpy()
        temp = temp[..., 1]
        roc_pre = np.append(roc_pre, temp)
        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
val_accurate = acc / test_num
times = np.array(times[1:])
times = np.mean(times)
print("time（ms）:", times / batch_size)
print('val_accuracy: %.3f' % val_accurate)
matrix = genConfusionMatrix(2, pre, tar)

matrix_se = matrix[1][1] / (matrix[1][0] + matrix[1][1])
matrix_sp = matrix[0][0] / (matrix[0][1] + matrix[0][0])
matrix_acc = (matrix[0][0] + matrix[1][1]) / np.sum(matrix)
matrix_pre = matrix[1][1] / (matrix[0][1] + matrix[1][1])
matrix_NPV = matrix[0][0] / (matrix[1][0] + matrix[0][0])
F1 = 2 * matrix[1][1] / (2 * matrix[1][1] + matrix[1][0] + matrix[0][1])
LR0 = (1 - matrix_se) / matrix_sp
LR1 = matrix_se / (1 - matrix_sp)
print("matrix：")
print(matrix)
print("matrix_se", matrix_se)
print("matrix_sp", matrix_sp)
print("matrix_acc", matrix_acc)
print("matric_pre/ppv", matrix_pre)
print("NPV", matrix_NPV)
print("F1", F1)
print("LR+", LR1)
print("LR-", LR0)
