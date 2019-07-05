#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【tf-rnn-mnist】
#
# 概要:
#      tensorflow を利用したMNIST(手書き文字認識)
#      Sequential MNIST (LSTM)を利用して推論します
#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
BATCH_SIZE = 50

# mnist データを格納したオブジェクトを作成
mnist = input_data.read_data_sets("./data/", one_hot = True)

# モデルの形状を設定
num_seq   = 28
num_input = 28

# 入力データの定義(入力データなので placeholder で定義する)
x = tf.placeholder(tf.float32, [None, 784])
# 入力情報へ reshape
# (バッチサイズ, 高さ, 横幅) に変換 [チャンネルは必要ないよ...]
input = tf.reshape(x, [-1, 28, 28])

#===
# ネットワークの定義
#===
#
# 入力(x)層から畳み込み層1
with tf.name_scope("LSTM"):
    # ユニット数 128 (出力) のLSTM セル
    stacked_cells = []
    for _ in range(3):
        stacked_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=128))
    # MultiRNNCellでラップする必要がある
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_cells)
    # 動的RNN による時間発展
    outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=input, dtype=tf.float32)
    # 最後の時間軸のTensor をスライスして取得
    last_output = outputs[:,-1,:]
    w = tf.Variable(tf.truncated_normal([128, 10], stddev = 0.1))
    b = tf.Variable(tf.zeros([10]))
    out = tf.nn.softmax(tf.matmul(last_output, w) + b)    

# 誤差関数の定義
y = tf.placeholder(tf.float32, [None, 10])
with tf.name_scope("loss"):
    # クロスエントロピー
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out + 1.e-5), axis=[1]))
    
# trainer オブジェクト
with tf.name_scope("trian"):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 精度評価
with tf.name_scope("accuracy"):
    correct  = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    
# 初期化
init = tf.global_variables_initializer()

# ログのマージ
summary_op = tf.summary.merge_all()

# 学習の実行
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="1", # specify GPU number
        allow_growth=True
    )
)
# モデル保存(saver)
saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    summary_writer = tf.summary.FileWriter("logs", sess.graph)
    sess.run(init)
    
    # テスト用の全画像データ (10000,10)
    test_images = mnist.test.images
    # テスト用の全正解データ (10000,10)
    test_labels = mnist.test.labels

    for i in range(1000):
        step = i + 1
        train_images, train_labels = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict = {x: train_images, y: train_labels})
        if step % 100 == 0:
            # ログを取る処理をする
            summary_str = sess.run(summary_op, feed_dict = {x: test_images, y: test_labels})
            summary_writer.add_summary(summary_str, step)
            acc_val = sess.run(accuracy,feed_dict = {x: test_images, y: test_labels})
            print('Step %d: accuracy = %.2f' % (step, acc_val))
            saver.save(sess, 'ckpt/my_model', global_step = step, write_meta_graph = False)
    # モデル保存
    saver.save(sess, 'ckpt/my_model')
