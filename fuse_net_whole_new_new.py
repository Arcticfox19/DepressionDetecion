#多模态模型的训练和测试
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import wave
import librosa
from python_speech_features import *
import re
#from allennlp.commands.elmo import ElmoEmbedder
import os
# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import itertools
from sklearn.model_selection import KFold
from answer_features_whole import get_text_target_new

prefix = os.path.abspath(os.path.join(os.getcwd(), "./"))

#text_features = np.load(os.path.join(prefix, 'Features/AnswerWhole/whole_samples_36x3.npz'))['arr_0']
# loaded_data = np.load(os.path.join(prefix, 'Features/AnswerWhole/whole_samples_36x3.npz'))
# text_features = list(loaded_data.values())
text_features_path='Mydata'
#text_features,text_targets=get_text_target(text_features_path)
text_features,text_targets=get_text_target_new(text_features_path)
#text_targets = np.load(os.path.join(prefix, 'Features/AnswerWhole/whole_labels_36x3.npz'))['arr_0']
audio_features = np.squeeze(np.load(os.path.join(prefix, 'Features/AudioWhole/whole_samples_36x3_256.npz'))['arr_0'], axis=2)
audio_targets = np.load(os.path.join(prefix, 'Features/AudioWhole/whole_labels_36x3_256.npz'))['arr_0']


fuse_features = [[audio_features[i], text_features[i]] for i in range(len(text_features))]
fuse_targets = text_targets


fuse_dep_idxs = np.where(text_targets == 1)[0]
fuse_non_idxs = np.where(text_targets == 0)[0]

def save(model, filename):
    save_filename = '{}.pt'.format(filename)
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)
    
def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    #输出预测和真实值的情况
    # [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    conf_matrix= confusion_matrix(y_test, y_test_pred)
    # 检查混淆矩阵的形状
    if conf_matrix.shape == (2, 2):
        [[tn, fp], [fn, tp]] = conf_matrix
    else:
        # 如果形状不是2x2，则手动获取混淆矩阵中的值
        tn = conf_matrix[0, 0]
        fp = 0
        fn = 0  # 因为没有阳性样本，所以假阴性为0
        tp = 0  # 因为没有阳性样本，所以真阳性为0
    return np.array([[tp, fp], [fn, tn]])

def model_performance(y_test, y_test_pred_proba):
    """
    Evaluation metrics for network performance.
    """
    y_test_pred = y_test_pred_proba.flatten()  # 将张量扁平化为一维数组

    conf_matrix = standard_confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    return y_test_pred, conf_matrix

class fusion_net(nn.Module):
    def __init__(self, text_embed_size, text_feature_size,rnn_layers, dropout, num_classes, \
                 audio_hidden_dims, audio_embed_size):
        super(fusion_net, self).__init__()
        self.text_embed_size = text_embed_size
        self.text_feature_size = text_feature_size
        self.audio_embed_size = audio_embed_size
        self.audio_hidden_dims = audio_hidden_dims
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.num_classes = num_classes

        # ============================= TextLogisticRegression =================================
        # self.text_model = TextLogisticRegression(self.text_embed_size, self.text_feature_size)
        # ============================= TextLogisticRegression =================================

        # ============================= AudioBiLSTM =============================

        self.lstm_net_audio = nn.GRU(self.audio_embed_size,
                                     self.audio_hidden_dims,
                                     num_layers=self.rnn_layers,
                                     dropout=self.dropout,
                                     bidirectional=False,
                                     batch_first=True)

        self.fc_audio = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.audio_hidden_dims, self.audio_hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.ln = nn.LayerNorm(self.audio_embed_size)

        # ============================= AudioBiLSTM =============================

        # ============================= last fc layer =============================
        # self.bn = nn.BatchNorm1d(self.text_hidden_dims + self.audio_hidden_dims)
        # modal attention
        self.modal_attn = nn.Linear(self.text_embed_size + 128,  # 修改输入大小为text_embed_size
                                    self.text_embed_size + 128, bias=False)
        self.fc_final = nn.Sequential(
            nn.Linear(self.text_embed_size + 128, self.num_classes, bias=False),
            # nn.ReLU(),
            nn.Softmax(dim=1),
            # nn.Sigmoid()
        )

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def pretrained_feature(self, x):
        with torch.no_grad():
            x_text = []
            x_audio = []
            for ele in x:
                x_text.append(ele[1])
                x_audio.append(ele[0])
            x_text, x_audio = Variable(torch.tensor(x_text).type(torch.FloatTensor), requires_grad=False), Variable(
                torch.tensor(x_audio).type(torch.FloatTensor), requires_grad=False)
            # ============================= TextBiLSTM =================================
            # x : [len_seq, batch_size, embedding_dim]
            # x_text = x_text.permute(1, 0, 2)

            #原版
            # text_output = self.text_model(x_text)
            # text_feature = text_output

            #直接使用归一化后的feature
            text_feature=x_text

            # ============================= TextBiLSTM =================================

            # ============================= AudioBiLSTM =============================
            x_audio = self.ln(x_audio)

            x_audio, _ = self.lstm_net_audio(x_audio)

            x_audio = x_audio.sum(dim=1)

            audio_feature = self.fc_audio(x_audio)

            audio_pooled = nn.AdaptiveAvgPool1d(128)(audio_feature)

            # audio_feature = expand_audio_feature(audio_feature)
            # print("Expanded audio_feature shape:", audio_feature.shape)
            # print('text_feature shape:',text_feature.shape)
            #print('audio_feature',audio_feature)


        # ============================= AudioBiLSTM =============================
        return (text_feature, audio_pooled)

    def forward(self, x):
        # x = self.bn(x)
        # modal_weights = torch.softmax(self.modal_attn(x), dim=1)
        # modal_weights = self.modal_attn(x)
        # x = (modal_weights * x)
        #print('x shape:',x.shape)
        output = self.fc_final(x)
        return output

def expand_audio_feature(audio_feature):
    # 假设传入的 audio_feature 的维度是 (batch_size, 2)
    # audio_feature = torch.tensor(audio_feature)  # 将列表转换为张量
    return audio_feature.unsqueeze(1)  # 在维度1处添加一个维

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, text_feature, audio_feature, target, model):
        weight = model.fc_final[0].weight
        pred_text = F.linear(text_feature, weight[:, :model.text_embed_size])  # 修改这里
        pred_audio = F.linear(audio_feature, weight[:, model.text_embed_size:])  # 修改这里
        l = nn.CrossEntropyLoss()
        # 将目标张量转换为 torch.long 类型
        target = torch.tensor(target, dtype=torch.long)
        return l(pred_text, target) + l(pred_audio, target)

config = {
    'num_classes': 2,
    'dropout': 0.5,
    'rnn_layers': 4,
    'audio_embed_size': 256,
    'audio_hidden_dims': 256,

    #'text_embed_size': 1024,
    #反应时
    'text_embed_size':13,
    'text_feature_size':13,
    'batch_size': 8,
    'epochs': 400,
    'learning_rate': 0.0005,
    #'text_hidden_dims': 128,
    'cuda': False,
    'lambda': 1e-5,
}

model = fusion_net(config['text_embed_size'],config['text_feature_size'],config['rnn_layers'], \
    config['dropout'], config['num_classes'], config['audio_hidden_dims'], config['audio_embed_size'])

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
# optimizer = optim.Adam(model.parameters())
# criterion = nn.CrossEntropyLoss()
criterion = MyLoss()

# def expand_audio_feature(audio_feature):
#     # 假设传入的 audio_feature 的维度是 (batch_size, 2)
#     #audio_feature = torch.tensor(audio_feature)  # 将列表转换为张量
#     return audio_feature.unsqueeze(1)  # 在维度1处添加一个维度

def train(epoch, train_idxs):
    global max_train_acc, train_acc
    model.train()
    batch_idx = 1
    total_loss = 0
    correct = 0
    X_train = []
    Y_train = []
    for idx in train_idxs:
        X_train.append(fuse_features[idx])
        Y_train.append(fuse_targets[idx])
    for i in range(0, len(X_train), config['batch_size']):
        if i + config['batch_size'] > len(X_train):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+config['batch_size'])], Y_train[i:(i+config['batch_size'])]
        if config['cuda']:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(torch.from_numpy(y)).cuda()
        # 将模型的参数梯度设置为0
        optimizer.zero_grad()
        text_feature, audio_feature = model.pretrained_feature(x)
        #print('audio_feature shape', audio_feature.shape)

        #对audio_feature归一化
        with torch.no_grad():
            # concat_x = torch.cat((audio_feature, text_feature), dim=1)
            audio_feature_norm = (audio_feature - audio_feature.mean())/audio_feature.std()
            concat_x = torch.cat((text_feature, audio_feature_norm), dim=1)
            #print('concat_x shape',concat_x.shape)
            output = model(concat_x)

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(torch.tensor(y).data.view_as(pred)).cpu().sum()
        # loss = criterion(output, torch.tensor(y))
        loss = criterion(text_feature, audio_feature, y, model)
        # 后向传播调整参数
        loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        batch_idx += 1
        # loss.item()能够得到张量中的元素值
        total_loss += loss.item()
    cur_loss = total_loss
    max_train_acc = correct
    train_acc = correct
    print('Train Epoch: {:2d}\t Learning rate: {:.4f}\tLoss: {:.6f}\t Accuracy: {}/{} ({:.0f}%)\n '.format(
                epoch, config['learning_rate'], cur_loss/len(X_train), correct, len(X_train),
        100. * correct / len(X_train)))


def evaluate(model, test_idxs, fold, train_idxs):
    model.eval()
    batch_idx = 1
    total_loss = 0
    pred = torch.empty(config['batch_size'], 1).type(torch.LongTensor)
    X_test = []
    Y_test = []
    for idx in test_idxs:
        X_test.append(fuse_features[idx])
        Y_test.append(fuse_targets[idx])
    global max_train_acc, max_acc,max_f1
    for i in range(0, len(X_test), config['batch_size']):
        if i + config['batch_size'] > len(X_test):
            x, y = X_test[i:], Y_test[i:]
        else:
            x, y = X_test[i:(i+config['batch_size'])], Y_test[i:(i+config['batch_size'])]
        if config['cuda']:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(torch.from_numpy(y)).cuda()
        text_feature, audio_feature = model.pretrained_feature(x)
        with torch.no_grad():
            audio_feature_norm = (audio_feature - audio_feature.mean())/audio_feature.std()
            text_feature_norm = (text_feature - text_feature.mean())/text_feature.std()
            concat_x = torch.cat((text_feature, audio_feature_norm), dim=1)
            output = model(concat_x)

        # loss = criterion(output, torch.tensor(y))
        loss = criterion(text_feature, audio_feature, y, model)
        pred = torch.cat((pred, output.data.max(1, keepdim=True)[1]))
        total_loss += loss.item()
        
    y_test_pred, conf_matrix = model_performance(Y_test, pred[config['batch_size']:])
    
    print('\nTest set: Average loss: {:.4f}'.format(total_loss/len(X_test)))
    # custom evaluation metrics
    print('Calculating additional test metrics...')
    accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1]) if (conf_matrix[0][0] + conf_matrix[0][1]) > 0 else 0
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0]) if (conf_matrix[0][0] + conf_matrix[1][0]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("Accuracy: {}".format(accuracy))
    #print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}\n".format(f1_score))
    print('='*89)
    
    if max_f1 < f1_score and max_train_acc >= len(train_idxs)*0.8 and f1_score > 0.5:
        max_f1 = f1_score
        max_acc = accuracy
        save(model, os.path.join(prefix, 'Model/ClassificationWhole/Fuse/fuse_{:.2f}_{}'.format(max_f1, fold)))
        print('*'*64)
        print('model saved: f1: {}\tacc: {}'.format(max_f1, max_acc))
        print('*'*64)
    return total_loss

if __name__ == '__main__':
    kf = KFold(n_splits=3, shuffle=True)
    fold = 1
    for train_idxs_tmp, test_idxs_tmp in kf.split(fuse_features):
        train_idxs, test_idxs = [], []
        resample_idxs = [0,1,2,3,4,5]

        # depression data augmentation
        for idx in train_idxs_tmp:
            if idx in fuse_dep_idxs:
                feat = fuse_features[idx]
                count = 0
                for i in itertools.permutations(feat, feat.shape[0]):
                    if count in resample_idxs:
                        fuse_features = np.vstack((fuse_features, np.expand_dims(list(i), 0)))
                        fuse_targets = np.hstack((fuse_targets, 1))
                        train_idxs.append(len(fuse_features)-1)
                    count += 1
            else:
                train_idxs.append(idx)

        for idx in test_idxs_tmp:
            if idx in fuse_dep_idxs:
                feat = fuse_features[idx]
                count = 0
                #resample_idxs = random.sample(range(6), 4)
                resample_idxs = [0,1,4,5]
                for i in itertools.permutations(feat, feat.shape[0]):
                    if count in resample_idxs:
                        fuse_features = np.vstack((fuse_features, np.expand_dims(list(i), 0)))
                        fuse_targets = np.hstack((fuse_targets, 1))
                        test_idxs.append(len(fuse_features)-1)
                    count += 1
            else:
                test_idxs.append(idx)

        #未增强
        #train_idxs, test_idxs = train_idxs_tmp.tolist(), test_idxs_tmp.tolist()

        # 定义数据集和验证集
        # train_idxs_tmp = np.load(os.path.join(prefix, 'Features/TextWhole/train_idxs_0.{}.npy'.format(fold)), allow_pickle=True)
        # test_idxs_tmp = list(set(list(fuse_dep_idxs) + list(fuse_non_idxs)) - set(train_idxs_tmp))

        for param in model.parameters():
            param.requires_grad = False

        model.fc_final[0].weight.requires_grad = True

        max_f1 = -1
        max_acc = -1
        max_train_acc = -1

        for ep in range(1, config['epochs']):
            train(ep, train_idxs)
            tloss = evaluate(model, test_idxs, fold, train_idxs)

        fold += 1