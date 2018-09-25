#win環境ではcsvファイルの文字コードはANSIである必要がある．

import tensorflow as tf
import numpy as np
import csv
import pandas as pd
import os

'''パラメタ'''
datanum = 6
labnum = 3
step_num = 1000
num_units = 10
batch_size = 100
'''csvファイルからデータの読み込み'''
#csvdataのかたちは以下のように
# 5.2, 6.1, 7.2, 8.2, 1(int)
# 2.3, 4.5, 6.7, 2.4, 2(int)
# ...
# data,data,data,data,label

def csv_loader(filename):
    csv_obj = csv.reader(open(filename, "r"))
    dt = [ v for v in csv_obj]
    dat = [[float(elm) for elm in v] for v in dt]
    db = [[0 for n in range(datanum)] for m in range(len(dat))]
    lb = [[0 for n in range(labnum)] for mm in range(len(dat))]
    for i in range(len(dat)):
        for j in range(len(dat[i])):
            if j <= datanum - 1:
                db[i][j] = dat[i][j]
            else:
                lb[i][j-datanum] = dat[i][j]
    return (db,lb)

data_body,label_body = csv_loader("analytical_data.csv")
data_test_body,label_test_body = csv_loader("test.csv")
#ndarrayに変換
data_body = np.array(data_body)
#print(data_body)
data_test_body = np.array(data_test_body)
label_body = np.array(label_body)
label_test_body = np.array(label_test_body)

def zscore(x):
    xmean = x.mean()
    xstd  = np.std(x)

    zscore = (x-xmean)/xstd
    return zscore

def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

#data_body = zscore(data_body)
#data_test_body = zscore(data_test_body)
#placeholder
data = tf.placeholder(dtype=tf.float32,shape=[None,datanum])
label = tf.placeholder(dtype=tf.float32,shape=[None,labnum])
keep_prob = tf.placeholder(tf.float32)

#隠れ層
b3 = tf.Variable(tf.zeros([num_units]))
w3 = tf.Variable(tf.truncated_normal([datanum, num_units]))
hidden3 = tf.nn.relu(tf.matmul(data,w3) + b3)
drop_out_h3 = tf.nn.dropout(hidden3, keep_prob)

b2 = tf.Variable(tf.zeros([num_units]))
w2 = tf.Variable(tf.truncated_normal([num_units, num_units]))
hidden2 = tf.nn.relu(tf.matmul(drop_out_h3,w2) + b2)
drop_out_h2 = tf.nn.dropout(hidden2, keep_prob)

b1 = tf.Variable(tf.zeros([num_units]))
w1 = tf.Variable(tf.truncated_normal([num_units, num_units]))
hidden1 = tf.nn.relu(tf.matmul(drop_out_h2,w1) + b1)
drop_out_h1 = tf.nn.dropout(hidden1, keep_prob)

b0 = tf.Variable(tf.zeros([labnum],dtype=tf.float32))
w0 = tf.Variable(tf.zeros([num_units,labnum],dtype=tf.float32))
y = tf.nn.leaky_relu(tf.matmul(drop_out_h1,w0) + b0)

def loss(y,label):
    cost = tf.reduce_mean(tf.square(y - label))
    return cost

cost = loss(y,label)

def training(loss):
    with tf.name_scope('training') as scope:
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    return train_step

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

logs = []
with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    #s.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    cwd = os.getcwd()

    for i in range(step_num):
        if i % 100 == 0:
            print(i)
        sff_idx = np.random.permutation(data_body.shape[0])

        # ミニバッチ
        for idx in range(0, data_body.shape[0], batch_size):
            batch_x = data_body[sff_idx[idx: idx+batch_size]]
            batch_l = label_body[sff_idx[idx: idx+batch_size]]
            s.run(train_step,feed_dict={data:batch_x, label:batch_l, keep_prob:0.95})

        #学習曲線を描くためのログ
        train_loss = s.run(cost,feed_dict={data:data_body, label:label_body, keep_prob:1.0})

        log = {'epoch': i, 'train_loss': train_loss}
        logs.append(log)

    # Save logs
    df = pd.DataFrame(logs)
    df.to_csv("log.csv", index=False)

    #Save model
    saver.save(s, cwd + "/model.ckpt")

    #学習結果に基づく出力
    y_vals = s.run(y, feed_dict={data: data_test_body, label: label_test_body, keep_prob: 1.0})
    np.savetxt('out.csv',y_vals,delimiter=',')
