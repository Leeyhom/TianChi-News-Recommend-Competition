# TianChi-News-Recommend-Competition

## 简要

赛题以新闻APP中的新闻推荐为背景，目的是**要求根据用户历史浏览点击新闻的数据信息预测用户未来的点击行为**。

## 数据情况

数据来源自某新闻APP平台的用户交互数据，包含有30万用户，近300万次点击，共36万多篇不同的新闻文章，同时还有每篇新闻对应的embedding向量表示。

训练数据为其中的20万用户的点击数据，剩下的数据被均分为两个测试集A，B。

## 算法

### Baseline算法

采用物品协同过滤算法召回，召回后选择分数高的5篇文章推荐给用户，若遇到数量不够的情况下选择热门文章进行替代。

基于物品的协同过滤算法中考虑了对热门物品及活跃用户的惩罚。
