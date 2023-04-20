import yaml
import os
import numpy as np
from utils import get_snapshot

# load config
file = open('config.yml', 'r', encoding="utf-8")
file_data = file.read()
config=yaml.load(file_data,Loader=yaml.FullLoader) 

# build path
base_path = os.path.join('./data/', config['dataset'])
raw_base_path = os.path.join(base_path, 'raw')
train_save_path = os.path.join(base_path, 'train.npy')
test_save_path = os.path.join(base_path, 'test.npy')

# load data

num = len(os.listdir(raw_base_path)) #os.listdir用于返回指定文件夹下包含的文件或文件夹的名字的列表
data = np.zeros(shape=(num, config['node_num'], config['node_num']), dtype=np.float32)#返回一个给定形状和类型的用0填充的数组
for i in range(num):
    path = os.path.join(raw_base_path, 'edge_list_' + str(i) + '.txt')
    data[i] = get_snapshot(path, config['node_num'], config['max_thres']) #每一个文件生成一个快照

total_num = num - config['window_size']
test_num = int(config['test_rate'] * total_num)
train_num = total_num - test_num

train_data = data[0: train_num + config['window_size']]
test_data = data[train_num: num]

# save data

np.save(train_save_path, train_data)
np.save(test_save_path, test_data)