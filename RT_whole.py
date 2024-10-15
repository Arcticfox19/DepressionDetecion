#对文本特征进行分类
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
# from allennlp.commands.elmo import ElmoEmbedder
import os
# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
import itertools
from sklearn.model_selection import KFold
from answer_features_whole import get_text_target_new

prefix = os.path.abspath(os.path.join(os.getcwd(), "./"))

text_features_path = 'Mydata'
text_features, text_targets = get_text_target_new(text_features_path)
# text_targets = np.load(os.path.join(prefix, 'Features/AnswerWhole/whole_labels_36x3.npz'))['arr_0']

dep_idxs = np.where(text_targets == 1)[0]
non_idxs = np.where(text_targets == 0)[0]


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
    # 输出预测和真实值的情况
    # print('y_test', y_test)
    # print('y_test_pred', y_test_pred)
    # [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
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

class TextLogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TextLogisticRegression, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output, target):
        criterion = nn.BCELoss()
        return criterion(output.float(), target.float())


# 在训练和测试函数中使用文本特征
def train(epoch, train_idxs):
    global max_train_acc, train_acc
    model.train()
    batch_idx = 1
    total_loss = 0
    correct = 0
    X_train = []
    Y_train = []
    for idx in train_idxs:
        X_train.append(text_features[idx])  # 只使用文本特征
        Y_train.append(text_targets[idx])
    for i in range(0, len(X_train), config['batch_size']):
        if i + config['batch_size'] > len(X_train):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i + config['batch_size'])], Y_train[i:(i + config['batch_size'])]
        # if config['cuda']:
        #     x, y = Variable(torch.from_numpy(np.array(x)).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(
        #         torch.from_numpy(np.array(y))).cuda()
        # else:
        #     x= Variable(torch.tensor(np.array(x)).type(torch.FloatTensor), requires_grad=False)
        #     y = torch.tensor(np.array(y)).type(torch.LongTensor)
        if config['cuda']:
            x, y = Variable(torch.tensor(np.array(x)).type(torch.FloatTensor).cuda(), requires_grad=True), Variable(
                torch.tensor(np.array(y)).type(torch.FloatTensor).cuda(), requires_grad=True)
        else:
            x, y = Variable(torch.tensor(np.array(x)).type(torch.FloatTensor), requires_grad=True), Variable(
                torch.tensor(np.array(y)).type(torch.FloatTensor), requires_grad=True)
        optimizer.zero_grad()
        optimizer.zero_grad()
        output = model(x)  # 只传入文本特征
        pred = output.data.max(1, keepdim=True)[1].squeeze()
        correct += pred.eq(torch.tensor(y).data.view_as(pred)).cpu().sum()
        print(pred.dtype)
        print(y.dtype)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()
    cur_loss = total_loss
    max_train_acc = correct
    train_acc = correct
    print('Train Epoch: {:2d}\t Learning rate: {:.4f}\tLoss: {:.6f}\t Accuracy: {}/{} ({:.0f}%)\n '.format(
        epoch, config['learning_rate'], cur_loss / len(X_train), correct, len(X_train),
                                        100. * correct / len(X_train)))

def evaluate(model, test_idxs, fold, train_idxs):
    model.eval()
    batch_idx = 1
    total_loss = 0
    pred = torch.empty(config['batch_size'], 1).type(torch.FloatTensor)
    #pred=[]
    X_test = []
    Y_test = []
    for idx in test_idxs:
        X_test.append(text_features[idx])  # 只使用文本特征
        Y_test.append(text_targets[idx])
    global max_train_acc, max_acc, max_f1
    for i in range(0, len(X_test), config['batch_size']):
        if i + config['batch_size'] > len(X_test):
            x, y = X_test[i:], Y_test[i:]
        else:
            x, y = X_test[i:(i + config['batch_size'])], Y_test[i:(i + config['batch_size'])]
        if config['cuda']:
            x, y = Variable(torch.from_numpy(np.array(x)).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(
                torch.from_numpy(np.array(y))).cuda()
        else:
            x, y = Variable(torch.tensor(np.array(x)).type(torch.FloatTensor), requires_grad=True), Variable(
                torch.tensor(np.array(y)).type(torch.FloatTensor), requires_grad=True)
        with torch.no_grad():
            output = model(x)  # 只传入文本特征
        loss = criterion(output.data.max(1, keepdim=True)[1].squeeze(), y)
        pred = torch.cat((pred, output.data.max(1, keepdim=True)[1]))
        total_loss += loss.item()

    y_test_pred, conf_matrix = model_performance(Y_test, pred[config['batch_size']:])

    print('\nTest set: Average loss: {:.4f}'.format(total_loss / len(X_test)))
    print('Calculating additional test metrics...')
    accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1]) if (conf_matrix[0][0] +
                                                                                       conf_matrix[0][1]) > 0 else 0
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0]) if (conf_matrix[0][0] + conf_matrix[1][
        0]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("Accuracy: {}".format(accuracy))
    print("Recall: {}".format(recall))
    print("F1-Score: {}\n".format(f1_score))
    print('=' * 89)

    return total_loss

config = {
    'num_classes': 2,
    'text_embed_size':13,
    'batch_size': 8,
    'epochs': 100,
    'learning_rate': 0.0005,
    'cuda': False,
    'lambda': 1e-5,
}
model = TextLogisticRegression(config['text_embed_size'],config['num_classes'])
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = MyLoss()

if __name__ == '__main__':
    kf = KFold(n_splits=3, shuffle=True)
    fold = 1
    for train_idxs_tmp, test_idxs_tmp in kf.split(text_features):
        # 未增强
        train_idxs, test_idxs = train_idxs_tmp.tolist(), test_idxs_tmp.tolist()

        # 初始化评估指标
        max_f1 = -1
        max_acc = -1
        max_train_acc = -1

        for param in model.parameters():
            param.requires_grad = False

        #model.fc_final[0].weight.requires_grad = True

        for ep in range(1, config['epochs']):
            train(ep, train_idxs)
            tloss = evaluate(model, test_idxs, fold, train_idxs)

        fold += 1
