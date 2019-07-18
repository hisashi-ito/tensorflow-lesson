#! /bin/bash
#
#【build】
#
# 概要: tensorflow-lesson 用の container をビルドする
#
sudo docker build -t tensor-flow -f ./Dockerfile .
