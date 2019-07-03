#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【tf-cnn-mnist】
#
# 概要:
#      tensorflow を利用したMNIST(手書き文字認識)
#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
BATCH_SIZE = 50

# mnist データを格納したオブジェクトを作成
mnist = input_data.read_data_sets("./data/", one_hot = True)

# 学習データの取得(バッチサイズ)
# train_images, train_labels = mnist.train.next_batch(BATCH_SIZE)
# (50, 784)
# print(train_images.shape) 

# テスト用の全画像データ (10000,10)
# test_images = mnist.test.images
# テスト用の全正解データ (10000,10)
# test_labels = mnist.test.labels

# (お勉強)
#
#  Variable と Placeholder の違い
#
#  [Variable]
#   ネットワークの入力層,隠れ層,出力層の重みやパラメータを定義,保存しておくためのデータ構造
#  [Placeholder]
#   ニューラルネットの入力として与えるデータの形を定義しておくためのデータ構造 
#
#  https://qiita.com/eve_yk/items/e42431200a1616c7d045
#  https://qiita.com/icoxfog417/items/fb5c24e35a849f8e2c5d
#  https://qiita.com/MENDY/items/49bff2c16d7a49243acd (関数の定義方法)

# 入力データの定義(入力データなので placeholder で定義する)
x = tf.placeholder(tf.float32, [None, 784])

# 入力画像
# (バッチサイズ, 高さ, 横幅, チャネル数) に変換
img = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image("input_data", img, 100)

#===
# ネットワークの定義
#===
#
# 入力(x)層から畳み込み層1
with tf.name_scope("hidden"):

    # フィルタの定義
    # (縦, 横, チャネル数, フィルタの枚数(畳み込み後のチャネル数))
    f1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev =0.1))
    # strides: (バッチ方向, 縦方向, 横方向, チャネル方向)
    # 留意点)
    #        バッチ方向, チャネル方向は 1 以外指定してはいけない。スキップされるので...
    conv1 = tf.nn.conv2d(img, f1, strides = [1,1,1,1], padding = 'SAME')
    # バイアス項がなぜ "0.1" で初期化している... 
    b1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.relu(conv1 + b1)
    # プーリング層
    # ksize: フィルタサイズ (バッチ方向, 縦方向, 横方法, チャネル方向)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = "SAME")

    # 畳み込み層2 (チャネル数を2倍)
    f2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev =0.1))
    conv2 = tf.nn.conv2d(h_pool1, f2, strides = [1,1,1,1], padding = 'SAME')
    b2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(conv2 + b2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = "SAME")

    # 全結合層
    # 画像7x7 で 64　チャネルあるネットワークを1次元構造してから 512へ結合
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    w_fc1 = tf.Variable(tf.truncated_normal([7* 7* 64, 512], stddev = 0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # 中間層の重みの分布をログ出力
    tf.summary.histogram("h_conv1", h_conv1)
    tf.summary.histogram("h_conv2", h_conv2)
    tf.summary.histogram("h_pool1", h_pool1)
    tf.summary.histogram("h_pool2", h_pool2)
    
# 中間層(h_1)から出力層(out)
with tf.name_scope("output"):
    w_fc2 = tf.Variable(tf.truncated_normal([512, 10], stddev = 0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    out = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

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

    for i in range(200):
        step = i + 1
        train_images, train_labels = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict = {x: train_images, y: train_labels})
        if step % 10 == 0:
            # ログを取る処理をする
            summary_str = sess.run(summary_op, feed_dict = {x: test_images, y: test_labels})
            summary_writer.add_summary(summary_str, step)
            acc_val = sess.run(accuracy,feed_dict = {x: test_images, y: test_labels})
            print('Step %d: accuracy = %.2f' % (step, acc_val))
    
    # モデル保存
    saver.save(sess, 'ckpt/my_model')
    
