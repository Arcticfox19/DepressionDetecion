## Acknowledgments

This project is developed based on the original work from ICASSP2022-Depression.

Original Project URL: https://github.com/speechandlanguageprocessing/ICASSP2022-Depression

#数据预处理
##问卷得分计算
vq文件用于计算问卷得分（直接从问卷星按文本导出即可），生成一个csv文件，以VQ_data.csv为例
#音频数据预处理
Detect_silence用于检测音频的静音片段

可以通过修改VR场景中录音文件的保存部分，直接将录音文件的名称统一设置为Negative0、1、2...
#模型训练
##多模态模型
fuse_net_whole_new_new实现了多模态模型的训练和测试

audio_features_whole用于将音频特征保存为.npz格式，在模型训练时直接读取

answer_features_whole用于处理反应时数据
##单模态
audio_gru_whole用GRU单模态模型对音频数据进行分类

RT_whole对文本特征进行分类
##数据展示平台
所展示的内容通过data_prepare文件生成，以answering_data.xlsx和audio_data.xlsx为例

也可以参考ICASSP2022-Depression-main这个文件中的代码

