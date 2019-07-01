#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【tf-mnist】
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
# 入力画像をログに出力
img = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image("input_data", img, 100)

#===
# ネットワークの定義
#===
#
# 入力(x)層から中間層(h_1)
with tf.name_scope("hidden"):
    w_1 = tf.Variable(tf.truncated_normal([784, 64], stddev = 0.1), name = "w1")
    b_1 = tf.Variable(tf.zeros([64]), name = "b1")
    h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1) # y = RELU((x * w) + b)
    # 中間層の重みの分布をログ出力
    tf.summary.histogram("w_1", w_1)
    
# 中間層(h_1)から出力層(out)
with tf.name_scope("output"):
    w_2 = tf.Variable(tf.truncated_normal([64, 10], stddev = 0.1), name = "w2")
    b_2 = tf.Variable(tf.zeros([10]), name = "b2")
    out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2)


y = tf.placeholder(tf.float32, [None, 10])
# 誤差関数の定義
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - out))
    tf.summary.scalar("loss", loss)
    
# trainer オブジェクト
with tf.name_scope("trian"):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

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
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("logs", sess.graph)
    sess.run(init)
    
    # テスト用の全画像データ (10000,10)
    test_images = mnist.test.images
    # テスト用の全正解データ (10000,10)
    test_labels = mnist.test.labels

    for i in range(10000):
        step = i + 1
        train_images, train_labels = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict = {x: train_images, y: train_labels})
        if step % 10 == 0:
            # ログを取る処理をする
            summary_str = sess.run(summary_op, feed_dict = {x: test_images, y: test_labels})
            summary_writer.add_summary(summary_str, step)
            acc_val = sess.run(accuracy,feed_dict = {x: test_images, y: test_labels})
            print('Step %d: accuracy = %.2f' % (step, acc_val))
