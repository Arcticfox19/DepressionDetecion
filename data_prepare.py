#这个文件用于合并所有子文件夹中的数据
import os
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt

def load_answering(folder_path):
    # 合并所有子文件夹中的数据
    combined_data = []
    id=0
    # combined_data.append(['Name','PaintingName','IsAnswerCorrect','ReactionTime'])  # 添加表头
    combined_data.append(['DataID','Name', 'PaintingName', 'IsAnswerCorrect', 'ReactionTime'])
    names = {
        "ZLL": "曾丽莉", "ZCC": "赵川川", "WSM": "魏思敏", "XJY": "薛静艳", "ZQG": "郑琪光", "THY": "田昊宇", "ZXW": "张新伍",
        "LY": "龙垚", "QH": "钱行", "XXC": "徐小策", "ZL": "张璐", "ZJQ": "赵家奇", "MZH": "麻肇航", "LTF": "李腾飞",
        "DY0": "杜洋", "JK": "江凯", "LJL": "刘佳玲", "WJD": "万金斗", "LQ": "李强", "ZH": "张恒", "YYH": "闫宇航", "WD": "文迪",
        "YRY": "尤儒彦", "ZCX": "赵辰羲", "SQF": "沈启凡", "LZY": "刘泽宇", "DXY": "戴芯瑜", "SZX": "舒梓心", "YDY": "鄢灯莹", "HJQ": "何嘉琪",
        "MWL": "毛维澜", "YJJ": "袁婧婕", "SXP": "宋新鹏", "WCY": "王崇宇", "WHY": "王鸿燕", "DY1": "段钰"
    }
    # 遍历文件夹
    for subdir, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            # 拼接子文件夹路径
            subfolder_path = os.path.join(subdir, dir_name)
            name = dir_name.strip().split('_')[0]  # 使用第一个部分作为文件夹名称
            name = names.get(name, "")
            # 合并子文件夹中的所有 CSV 文件
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                # 检查文件扩展名是否为 .csv
                if file.endswith('.csv') and file != 'combined_data.csv':
                    # 读取 CSV 文件，忽略表头
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        # 移除表头
                        lines = lines[1:]
                        # 将 "isCorrectAnswer" 列的值转换为 '正确' 或 '错误'
                        for line in lines:
                            items = line.strip().split(',')
                            items[1] = '正确' if items[1] == "True" else '错误'
                            combined_data.append([id,name, items[0], items[1], items[2]])
                    id=id+1
    # 将合并后的数据存储到一个新的 DataFrame 中
    combined_df = pd.DataFrame(combined_data[1:], columns=combined_data[0])
    # 保存合并后的数据到一个新的 Excel 文件中的一个工作表中
    combined_file_path = os.path.join(folder_path, "../answering_data.xlsx")
    with pd.ExcelWriter(combined_file_path) as writer:
        combined_df.to_excel(writer, index=False, sheet_name='CombinedData')

def load_audio(folder_path):
    # 合并所有子文件夹中的数据
    combined_data = []
    id=0
    #combined_data.append(['Name','PaintingName','IsAnswerCorrect','ReactionTime'])  # 添加表头
    combined_data.append(['DataID','Name', 'PaintingName', 'AudioUrl', 'AudioWave'])
    painting_names = {
        "Neutral0.wav": "Modular_painting_with_four_panels",
        "Neutral1.wav": "House_on_the_Water",
        "Neutral2.wav": "Abstract_Composition"
    }
    names={
    "ZLL":"曾丽莉",    "ZCC":"赵川川",    "WSM":"魏思敏",    "XJY":"薛静艳",    "ZQG":"郑琪光","THY":"田昊宇","ZXW":"张新伍",
    "LY":"龙垚" ,"QH":"钱行","XXC":"徐小策","ZL":"张璐","ZJQ":"赵家奇","MZH":"麻肇航","LTF":"李腾飞",
    "DY0": "杜洋","JK":"江凯","LJL":"刘佳玲","WJD":"万金斗","LQ": "李强","ZH":"张恒","YYH":"闫宇航","WD":"文迪",
    "YRY": "尤儒彦","ZCX":"赵辰羲","SQF":"沈启凡","LZY":"刘泽宇","DXY": "戴芯瑜","SZX":"舒梓心","YDY":"鄢灯莹","HJQ":"何嘉琪",
    "MWL": "毛维澜","YJJ":"袁婧婕","SXP":"宋新鹏","WCY":"王崇宇","WHY": "王鸿燕","DY1":"段钰"
    }
    # 遍历文件夹
    for subdir, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            # 拼接子文件夹路径
            subfolder_path = os.path.join(subdir, dir_name)
            name = dir_name.strip().split('_')[0]  # 使用第一个部分作为文件夹名称
            name=names.get(name, "")
            # 合并子文件夹中的所有 CSV 文件
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                # 检查文件名
                if file=='Neutral0.wav'or file=='Neutral1.wav' or file=='Neutral2.wav':
                    painting_name = painting_names.get(file, "")  # 获取绘画名称
                    audio_url = file_path  # 获取音频文件路径
                    # 绘制波形图并保存
                    audio_wave_path = os.path.join(subfolder_path, file[:-4] + "_waveform.png")  # 保存波形图的路径
                    # 替换所有的反斜杠为斜杠
                    y, sr = librosa.load(file_path)
                    plt.figure(figsize=(10, 4))
                    plt.plot(y)
                    plt.savefig(audio_wave_path)
                    plt.close()
                    audio_url = os.path.join('../../static', file_path)
                    audio_url=audio_url.replace("\\", "/")
                    audio_wave_path = os.path.join('../../static', audio_wave_path)
                    audio_wave_path=audio_wave_path.replace("\\", "/")
                    # 添加新的数据
                    combined_data.append([id,name, painting_name, audio_url, audio_wave_path])
                id=id+1
    # 将合并后的数据存储到一个新的 DataFrame 中
    combined_df = pd.DataFrame(combined_data[1:], columns=combined_data[0])
    # 保存合并后的数据到一个新的 Excel 文件中的一个工作表中
    combined_file_path = os.path.join(folder_path, "../audio_data.xlsx")
    with pd.ExcelWriter(combined_file_path) as writer:
        combined_df.to_excel(writer, index=False, sheet_name='CombinedData')


if __name__ == '__main__':
    # 定义文件夹路径
    folder_path = "Mydata"
    load_answering(folder_path)
    load_audio(folder_path)
