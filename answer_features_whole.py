#用于处理反应时数据
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

def get_target(target):
    #将分数转化为0/1
    #EATD-Corpus数据集的转化方法为，分数小于53为0，大于等于53为1
    #本数据集的转化方法为，分数小于14为0，大于等于14为1，仅使用抑郁得分进行分类
    if target<14:
        return 0
    else:
        return 1

def extract_features_new(text_features, text_targets, path):
    # 初始化 MinMaxScaler 对象
    for subdir, dirs, files in os.walk(path):
        for dir_name in dirs:
            # 拼接子文件夹路径
            subfolder_path = os.path.join(subdir, dir_name)

            features = []
            targets = []
            with open(os.path.join(subfolder_path, 'combined_data.csv'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) == 2:  # Ensure each line contains exactly two parts
                        feature1, feature2 = map(float, parts)
                        features.append(feature2)
            with open(os.path.join(subfolder_path, 'new_label.txt')) as fli:
                target = float(fli.readline())
                target = get_target(target)
            # 每个被试的样本拆分成三条
            for i in range(0, 3):
                text_targets.append(target)
                text_features.append(features)  # Convert features list to numpy array
    # 初始化MinMaxScaler对象
    scaler = MinMaxScaler()
    # 对数据进行归一化
    #print('features_per_dimension',text_features)
    normalized_data = scaler.fit_transform(text_features)
    # 输出归一化后的数据
    #print('text_features_normalized:', normalized_data)
    print('text_features_normalized shape', normalized_data.shape)


#数据归一化
def get_text_target_new(path):
    text_features = []
    text_targets = []
    extract_features_new(text_features, text_targets, path)
    # print("Saving npz file locally...")
    # np.savez(os.path.join(prefix, 'Features/AnswerWhole/whole_samples_36x3.npz'), *text_features)
    # np.savez(os.path.join(prefix, 'Features/AnswerWhole/whole_labels_36x3.npz'), text_targets)
    # print('text_features',text_features)
    # print('text_targets',len(text_targets))
    return text_features,text_targets


