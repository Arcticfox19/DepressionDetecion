import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# 存储所有样本的反应时间和问卷分数
all_reaction_times = []
all_questionnaire_scores = []

# 遍历所有文件夹
for folder_name in os.listdir('./Mydata'):
    folder_path = os.path.join('./Mydata', folder_name)
    if os.path.isdir(folder_path):
        # 读取反应时间数据
        reaction_time_file = os.path.join(folder_path, 'combined_data.csv')
        reaction_time_data = pd.read_csv(reaction_time_file, header=None, sep=',')
        reaction_times = reaction_time_data.iloc[:, 1]  # 提取第二列的反应时间数据

        # 计算反应时间数据的标准差和均值
        reaction_time_std = reaction_times.std()
        reaction_time_mean = reaction_times.mean()

        # 定义阈值，通常选择均值加减一个或多个标准差
        threshold =reaction_time_std

        # 去除异常值
        filtered_reaction_times = reaction_times[abs(reaction_times - reaction_time_mean) <= threshold]

        # 读取问卷分数数据
        questionnaire_score_file = os.path.join(folder_path, 'new_label.txt')
        with open(questionnaire_score_file, 'r') as file:
            questionnaire_score = int(file.readline().strip())


        for reaction_time in filtered_reaction_times:
            all_reaction_times.append(reaction_time)
        all_questionnaire_scores.extend([questionnaire_score] * len(filtered_reaction_times))




# 计算皮尔逊相关系数
correlation_coefficient, _ = pearsonr(all_reaction_times, all_questionnaire_scores)
# 打印相关系数
print("反应时间和问卷分数的皮尔逊相关系数：", correlation_coefficient)
