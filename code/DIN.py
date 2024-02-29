import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import time
from datetime import datetime
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from sklearn.preprocessing import MinMaxScaler
import warnings

# 导入deepctr
from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from keras.utils import pad_sequences

from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import tensorflow as tf

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

warnings.filterwarnings("ignore")


# -----------------------返回排序后的结果-----------------------
def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=["user_id", "pred_score"])
    recall_df["rank"] = recall_df.groupby(["user_id"])["pred_score"].rank(ascending=False, method="first")

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby("user_id").apply(lambda x: x["rank"].max())
    assert tmp.min() >= topk

    del recall_df["pred_score"]
    submit = recall_df[recall_df["rank"] <= topk].set_index(["user_id", "rank"])
    submit = recall_df[recall_df["rank"] <= topk].set_index(["user_id", "rank"]).unstack(-1).reset_index()

    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(
        columns={"": "user_id", 1: "article_1", 2: "article_2", 3: "article_3", 4: "article_4", 5: "article_5"}
    )

    save_name = save_path + "/" + model_name + "_" + datetime.today().strftime("%m-%d") + ".csv"
    submit.to_csv(save_name, index=False, header=True)


# 排序结果归一化
def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


def get_kfold_users(trn_df, n=5):
    user_ids = trn_df["user_id"].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


# 数据准备函数
def get_din_feats_columns(df, dense_fea, sparse_fea, behavior_fea, hist_behavior_fea, emb_dim=32, max_len=100):
    """
    数据准备函数:
    df: 数据集
    dense_fea: 数值型特征列
    sparse_fea: 离散型特征列
    behavior_fea: 用户的候选行为特征列
    his_behavior_fea: 用户的历史行为特征列
    embedding_dim: embedding的维度， 这里为了简单， 统一把离散型特征列采用一样的隐向量维度
    max_len: 用户序列的最大长度
    """

    sparse_feature_columns = [
        SparseFeat(feat, vocabulary_size=df[feat].nunique() + 1, embedding_dim=emb_dim) for feat in sparse_fea
    ]

    dense_feature_columns = [
        DenseFeat(
            feat,
            1,
        )
        for feat in dense_fea
    ]

    var_feature_columns = [
        VarLenSparseFeat(
            SparseFeat(
                feat,
                vocabulary_size=df["click_article_id"].nunique() + 1,
                embedding_dim=emb_dim,
                embedding_name="click_article_id",
            ),
            maxlen=max_len,
        )
        for feat in hist_behavior_fea
    ]

    dnn_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns

    # 建立x, x是一个字典的形式
    x = {}
    for name in get_feature_names(dnn_feature_columns):
        if name in hist_behavior_fea:
            # 这是历史行为序列
            his_list = [l for l in df[name]]
            x[name] = pad_sequences(his_list, maxlen=max_len, padding="post")  # 二维数组
        else:
            x[name] = df[name].values

    return x, dnn_feature_columns


if __name__ == "__main__":
    data_path = "../data"
    feature_engineering_path = "../feature_engineering_result"
    save_path = "../final_result"
    offline = True

    # ---------------------------读取特征工程的数据-------------------------------
    # 重新读取数据的时候，发现click_article_id是一个浮点数，所以将其转换成int类型
    trn_user_item_feats_df = pd.read_csv(feature_engineering_path + "/trn_user_item_feats_df.csv")
    trn_user_item_feats_df["click_article_id"] = trn_user_item_feats_df["click_article_id"].astype(int)

    if offline:
        val_user_item_feats_df = pd.read_csv(feature_engineering_path + "/val_user_item_feats_df.csv")
        val_user_item_feats_df["click_article_id"] = val_user_item_feats_df["click_article_id"].astype(int)
    else:
        val_user_item_feats_df = None

    tst_user_item_feats_df = pd.read_csv(feature_engineering_path + "/tst_user_item_feats_df.csv")
    tst_user_item_feats_df["click_article_id"] = tst_user_item_feats_df["click_article_id"].astype(int)

    # 做特征的时候为了方便，给测试集也打上了一个无效的标签，这里直接删掉就行
    del tst_user_item_feats_df["label"]

    if offline:
        all_data = pd.read_csv("../data/train_click_log.csv")
        tst_data = pd.read_csv("../data/testA_click_log.csv")
        all_data = all_data._append(tst_data)
    else:
        trn_data = pd.read_csv("../data/train_click_log.csv")
        tst_data = pd.read_csv("../data/testA_click_log.csv")
        all_data = trn_data._append(tst_data)

    hist_click = all_data[["user_id", "click_article_id"]].groupby("user_id").agg({list}).reset_index()
    his_behavior_df = pd.DataFrame()
    his_behavior_df["user_id"] = hist_click["user_id"]
    his_behavior_df["hist_click_article_id"] = hist_click["click_article_id"]

    trn_user_item_feats_df_din_model = trn_user_item_feats_df.copy()

    if offline:
        val_user_item_feats_df_din_model = val_user_item_feats_df.copy()
    else:
        val_user_item_feats_df_din_model = None

    tst_user_item_feats_df_din_model = tst_user_item_feats_df.copy()

    trn_user_item_feats_df_din_model = trn_user_item_feats_df_din_model.merge(his_behavior_df, on="user_id")

    if offline:
        val_user_item_feats_df_din_model = val_user_item_feats_df_din_model.merge(his_behavior_df, on="user_id")
    else:
        val_user_item_feats_df_din_model = None

    tst_user_item_feats_df_din_model = tst_user_item_feats_df_din_model.merge(his_behavior_df, on="user_id")

    # 把特征分开
    sparse_fea = [
        "user_id",
        "click_article_id",
        "category_id",
        "click_environment",
        "click_deviceGroup",
        "click_os",
        "click_country",
        "click_region",
        "click_referrer_type",
        "is_cat_hab",
    ]

    behavior_fea = ["click_article_id"]

    hist_behavior_fea = ["hist_click_article_id"]

    dense_fea = [
        "sim0",
        "time_diff0",
        "word_diff0",
        "sim_max",
        "sim_min",
        "sim_sum",
        "sim_mean",
        "score",
        "rank",
        "click_size",
        "time_diff_mean",
        "active_level",
        "user_time_hob1",
        "user_time_hob2",
        "words_hbo",
        "words_count",
    ]

    # dense特征进行归一化, 神经网络训练都需要将数值进行归一化处理
    mm = MinMaxScaler()

    # 下面是做一些特殊处理，当在其他的地方出现无效值的时候，不处理无法进行归一化，刚开始可以先把他注释掉，在运行了下面的代码
    # 之后如果发现报错，应该先去想办法处理如何不出现inf之类的值
    # trn_user_item_feats_df_din_model.replace([np.inf, -np.inf], 0, inplace=True)
    # tst_user_item_feats_df_din_model.replace([np.inf, -np.inf], 0, inplace=True)

    for feat in dense_fea:
        trn_user_item_feats_df_din_model[feat] = mm.fit_transform(trn_user_item_feats_df_din_model[[feat]])

        if val_user_item_feats_df_din_model is not None:
            val_user_item_feats_df_din_model[feat] = mm.fit_transform(val_user_item_feats_df_din_model[[feat]])

        tst_user_item_feats_df_din_model[feat] = mm.fit_transform(tst_user_item_feats_df_din_model[[feat]])

    # 准备训练数据
    x_trn, dnn_feature_columns = get_din_feats_columns(
        trn_user_item_feats_df_din_model, dense_fea, sparse_fea, behavior_fea, hist_behavior_fea, max_len=50
    )
    y_trn = trn_user_item_feats_df_din_model["label"].values

    if offline:
        # 准备验证数据
        x_val, _ = get_din_feats_columns(
            val_user_item_feats_df_din_model, dense_fea, sparse_fea, behavior_fea, hist_behavior_fea, max_len=50
        )
        y_val = val_user_item_feats_df_din_model["label"].values

    dense_fea = [x for x in dense_fea if x != "label"]
    x_tst, _ = get_din_feats_columns(
        tst_user_item_feats_df_din_model, dense_fea, sparse_fea, behavior_fea, hist_behavior_fea, max_len=50
    )

    # 建立模型
    model = DIN(dnn_feature_columns, behavior_fea)

    # 查看模型结构
    model.summary()

    # 模型编译
    model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", tf.keras.metrics.AUC()])

    # 模型训练
    if offline:
        history = model.fit(x_trn, y_trn, verbose=1, epochs=10, validation_data=(x_val, y_val), batch_size=256)
    else:
        # 也可以使用上面的语句用自己采样出来的验证集
        # history = model.fit(x_trn, y_trn, verbose=1, epochs=3, validation_split=0.3, batch_size=256)
        history = model.fit(x_trn, y_trn, verbose=1, epochs=2, batch_size=256)

        # 模型预测
    tst_user_item_feats_df_din_model["pred_score"] = model.predict(x_tst, verbose=1, batch_size=256)
    tst_user_item_feats_df_din_model[["user_id", "click_article_id", "pred_score"]].to_csv(
        save_path + "/din_rank_score.csv", index=False
    )

    # 预测结果重新排序, 及生成提交结果
    rank_results = tst_user_item_feats_df_din_model[["user_id", "click_article_id", "pred_score"]]
    submit(rank_results, topk=5, model_name="din")
