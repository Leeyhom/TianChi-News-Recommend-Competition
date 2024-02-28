import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os, math, warnings, math, pickle
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model

from deepctr.feature_column import SparseFeat, VarLenSparseFeat

from keras.preprocessing.sequence import pad_sequences

from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
from collections import Counter

warnings.filterwarnings("ignore")


# ------------------ 读取数据 ---------------------
# debug模式： 从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path, sample_nums=10000):
    """
    训练集中采样一部分数据调试

    :param data_path: 原数据的存储路径
    :param sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """

    all_click = pd.read_csv(data_path + "train_click_log.csv")
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click["user_id"].isin(sample_user_ids)]

    all_click = all_click.drop_duplicates(
        (["user_id", "click_article_id", "click_timestamp"])
    )
    return all_click


# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path="./data_raw/", offline=True):
    if offline:
        all_click = pd.read_csv(data_path + "/train_click_log.csv")
    else:
        trn_click = pd.read_csv(data_path + "/train_click_log.csv")
        tst_click = pd.read_csv(data_path + "/testA_click_log.csv")
        all_click = trn_click._append(tst_click)

    all_click = all_click.drop_duplicates(
        (["user_id", "click_article_id", "click_timestamp"])
    )
    return all_click


# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df = pd.read_csv(data_path + "/articles.csv")

    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={"article_id": "click_article_id"})

    return item_info_df


# 读取文章的Embedding数据
def get_item_emb_dict(data_path, save_path="../multi_recall_result/"):
    item_emb_df = pd.read_csv(data_path + "/articles_emb.csv")

    item_emb_cols = [x for x in item_emb_df.columns if "emb" in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    # 进行归一化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    item_emb_dict = dict(zip(item_emb_df["article_id"], item_emb_np))
    pickle.dump(item_emb_dict, open(save_path + "item_content_emb.pkl", "wb"))

    return item_emb_dict


# --------------------工具函数
# 获取用户-文章-时间函数
# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):
    click_df = click_df.sort_values("click_timestamp")

    def make_item_time_pair(df):
        return list(zip(df["click_article_id"], df["click_timestamp"]))

    user_item_time_df = (
        click_df.groupby("user_id")[["click_article_id", "click_timestamp"]]
        .apply(lambda x: make_item_time_pair(x))
        .reset_index()
        .rename(columns={0: "item_time_list"})
    )
    user_item_time_dict = dict(
        zip(user_item_time_df["user_id"], user_item_time_df["item_time_list"])
    )

    return user_item_time_dict


# 根据时间获取商品被点击的用户序列  {item1: [(user1, time1), (user2, time2)...]...}
# 这里的时间是用户点击当前商品的时间，好像没有直接的关系。
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df["user_id"], df["click_timestamp"]))

    click_df = click_df.sort_values("click_timestamp")
    item_user_time_df = (
        click_df.groupby("click_article_id")[["user_id", "click_timestamp"]]
        .apply(lambda x: make_user_time_pair(x))
        .reset_index()
        .rename(columns={0: "user_time_list"})
    )

    item_user_time_dict = dict(
        zip(item_user_time_df["click_article_id"], item_user_time_df["user_time_list"])
    )
    return item_user_time_dict


# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=["user_id", "click_timestamp"])
    click_last_df = all_click.groupby("user_id").tail(1)

    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby("user_id").apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df


# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用
def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    item_info_df["created_at_ts"] = item_info_df[["created_at_ts"]].apply(
        max_min_scaler
    )

    item_type_dict = dict(
        zip(item_info_df["click_article_id"], item_info_df["category_id"])
    )
    item_words_dict = dict(
        zip(item_info_df["click_article_id"], item_info_df["words_count"])
    )
    item_created_time_dict = dict(
        zip(item_info_df["click_article_id"], item_info_df["created_at_ts"])
    )

    return item_type_dict, item_words_dict, item_created_time_dict


# 获取用户历史点击的文章信息
def get_user_hist_item_info_dict(all_click):
    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_typs = (
        all_click.groupby("user_id")["category_id"].agg(set).reset_index()
    )
    user_hist_item_typs_dict = dict(
        zip(user_hist_item_typs["user_id"], user_hist_item_typs["category_id"])
    )

    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict = (
        all_click.groupby("user_id")["click_article_id"].agg(set).reset_index()
    )
    user_hist_item_ids_dict = dict(
        zip(
            user_hist_item_ids_dict["user_id"],
            user_hist_item_ids_dict["click_article_id"],
        )
    )

    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words = (
        all_click.groupby("user_id")["words_count"].agg("mean").reset_index()
    )
    user_hist_item_words_dict = dict(
        zip(user_hist_item_words["user_id"], user_hist_item_words["words_count"])
    )

    # 获取user_id对应的用户最后一次点击的文章的创建时间
    all_click_ = all_click.sort_values("click_timestamp")
    user_last_item_created_time = (
        all_click_.groupby("user_id")["created_at_ts"]
        .apply(lambda x: x.iloc[-1])
        .reset_index()
    )

    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    user_last_item_created_time["created_at_ts"] = user_last_item_created_time[
        ["created_at_ts"]
    ].apply(max_min_scaler)

    user_last_item_created_time_dict = dict(
        zip(
            user_last_item_created_time["user_id"],
            user_last_item_created_time["created_at_ts"],
        )
    )

    return (
        user_hist_item_typs_dict,
        user_hist_item_ids_dict,
        user_hist_item_words_dict,
        user_last_item_created_time_dict,
    )


# 获取近期点击最多的k篇文章
def get_item_topk_click(click_df, k):
    topk_click = click_df["click_article_id"].value_counts().index[:k]
    return topk_click


# ------------------------召回评估函数
# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    last_click_item_dict = dict(
        zip(trn_last_click_df["user_id"], trn_last_click_df["click_article_id"])
    )
    user_num = len(user_recall_items_dict)

    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1

        hit_rate = round(hit_num * 1.0 / user_num, 5)

        print(
            "topK: {}, hit_num: {}, hit_rate: {}, user_num: {}".format(
                k, hit_num, hit_rate, user_num
            )
        )


# ----------------------------计算相似性矩阵
# 借鉴了KDD2020的去偏商品推荐，在计算item2item相似性矩阵时，使用关联规则，使得计算的文章相似性还考虑到了：
#   1. 用户点击的时间权重
#   2. 用户点击的顺序权重
#   3. 文章创建的时间权重
def itemcf_sim(df, item_created_time_dict):
    """
    文章与文章之间的相似性矩阵计算

    :param df: 数据表
    :param item_created_time_dict: 文章创建时间的字典
    :return: 文章与文章的相似性矩阵
    """

    user_item_time_dict = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)

    # 从同一用户的点击序列中，两两取出两篇文章，更新文章相似度矩阵
    print("\n构建文章相似度矩阵...")
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue

                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置信息权重，其中的参数可以调节
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间权重，其中的参数可以调节
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重，其中的参数可以调节
                created_time_weight = np.exp(
                    0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j])
                )
                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i][j] += (
                        loc_weight
                        * click_time_weight
                        * created_time_weight
                        / math.log(len(item_time_list) + 1)
                )

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + "/itemcf_i2i_sim.pkl", "wb"))

    return i2i_sim_


# ------------------------------------------------------usercf u2u_sim
def get_user_activate_degree_dict(all_click_df):
    """
    求用户活跃度
    """
    all_click_df_ = (
        all_click_df.groupby("user_id")["click_article_id"].count().reset_index()
    )

    # 用户活跃度归一化
    mm = MinMaxScaler()
    all_click_df_["click_article_id"] = mm.fit_transform(
        all_click_df_[["click_article_id"]]
    )
    user_activate_degree_dict = dict(
        zip(all_click_df_["user_id"], all_click_df_["click_article_id"])
    )

    return user_activate_degree_dict


def usercf_sim(all_click_df, user_activate_degree_dict):
    """
    用户相似性矩阵计算
    :param all_click_df: 数据表
    :param user_activate_degree_dict: 用户活跃度的字典
    :return: 用户相似性矩阵
    """

    item_user_time_dict = get_item_user_time_dict(all_click_df)

    u2u_sim = {}
    user_cnt = defaultdict(int)

    # 找到点击了同一篇文章的用户，计算相似度
    print("\n构建用户相似度矩阵...")
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, click_time in user_time_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                # 用户平均活跃度作为活跃度的权重，这里的式子也可以改善
                activate_weight = (
                        100
                        * 0.5
                        * (user_activate_degree_dict[u] + user_activate_degree_dict[v])
                )
                u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)

    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

    # 将得到的相似性矩阵保存到本地
    pickle.dump(u2u_sim_, open(save_path + "usercf_u2u_sim.pkl", "wb"))

    return u2u_sim_


# -------------------------item embedding sim
# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item
def embdding_sim(click_df, item_emb_df, save_path, topk):
    """
    基于内容的文章embedding相似性矩阵计算
    :param click_df: 数据表
    :param item_emb_df: 文章的embedding
    :param save_path: 保存路径
    :patam topk: 找最相似的topk篇
    :return: 文章相似性矩阵

    思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """

    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df["article_id"]))

    item_emb_cols = [x for x in item_emb_df.columns if "emb" in x]
    item_emb_np = np.ascontiguousarray(
        item_emb_df[item_emb_cols].values, dtype=np.float32
    )
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    # 建立faiss索引
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_index.search(item_emb_np, topk)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    item_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(
            zip(range(len(item_emb_np)), sim, idx)
    ):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            item_sim_dict[target_raw_id][rele_raw_id] = (
                    item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value
            )

    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(save_path + "/emb_i2i_sim.pkl", "wb"))

    return item_sim_dict


# --------------------YoutubeDNN召回
# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量
def gen_data_set(data, negsample=0):
    data.sort_values("click_timestamp", inplace=True)
    item_ids = data["click_article_id"].unique()

    train_set = []
    test_set = []

    # 获取训练数据和测试数据
    print("\n获取训练数据和测试数据...")
    for reviewerID, hist in tqdm(data.groupby("user_id")):
        pos_list = hist["click_article_id"].tolist()

        if negsample > 0:
            candidate_set = list(
                set(item_ids) - set(pos_list)
            )  # 用户没看过的文章里面选择负样本
            neg_list = np.random.choice(
                candidate_set, size=len(pos_list) * negsample, replace=True
            )  # 对于每个正样本，选择n个负样本

        # 长度只有一个的时候，需要把这条数据也放到训练集中，不然的话最终学到的embedding就会有缺失
        if len(pos_list) == 1:
            train_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list)))
            test_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list)))

        # 滑窗构造正负样本
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]

            # 倒数第一个之前的所有点击都是用户的历史点击，把训练集扩充
            if i != len(pos_list) - 1:
                train_set.append(
                    (reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]))
                )  # 正样本 [user_id, his_item, pos_item, label, len(his_item)]
                for negi in range(negsample):
                    train_set.append(
                        (
                            reviewerID,
                            hist[::-1],
                            neg_list[i * negsample + negi],
                            0,
                            len(hist[::-1]),
                        )
                    )  # 负样本 [user_id, his_item, neg_item, label, len(his_item)]
            else:
                # 将最长的那一个序列长度作为测试数据
                test_set.append(
                    (reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]))
                )

    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set


# 将输入的数据进行padding，使得序列特征的长度都一致
def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])

    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(
        train_seq, maxlen=seq_max_len, padding="post", truncating="post", value=0
    )
    train_model_input = {
        "user_id": train_uid,
        "click_article_id": train_iid,
        "hist_article_id": train_seq_pad,
        "hist_len": train_hist_len,
    }

    return train_model_input, train_label


# -------------------------------------------定义YouTubeDNN模型
def youtubednn_u2i_dict(data, save_path, topk=20):
    sparse_features = ["click_article_id", "user_id"]
    SEQ_LEN = 30  # 用户点击序列的长度，短的填充，长的截断

    user_profile_ = data[["user_id"]].drop_duplicates("user_id")
    item_profile_ = data[["click_article_id"]].drop_duplicates("click_article_id")

    # 类别编码
    features = ["click_article_id", "user_id"]
    feature_max_idx = {}

    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
        feature_max_idx[feature] = data[feature].max() + 1

    # 提取user和item的画像，这里具体选择哪些特征还需要进一步的分析和考虑
    user_profile = data[["user_id"]].drop_duplicates("user_id")
    item_profile = data[["click_article_id"]].drop_duplicates("click_article_id")

    user_index_2_rawid = dict(zip(user_profile["user_id"], user_profile_["user_id"]))
    item_index_2_rawid = dict(
        zip(item_profile["click_article_id"], item_profile_["click_article_id"])
    )

    # 划分训练和测试集
    # 由于深度学习需要的数据量通常都是非常大的，所以为了保证召回的效果，往往会通过滑窗的形式扩充训练样本
    # 每一个正样本，对应5个负样本
    train_set, test_set = gen_data_set(data, negsample=3)
    # 整理输入数据，具体的操作可以看上面的函数
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 确定Embedding的维度
    embedding_dim = 16

    # 将数据整理成模型可以直接输入的形式
    user_feature_columns = [
        SparseFeat("user_id", feature_max_idx["user_id"], embedding_dim),
        VarLenSparseFeat(
            SparseFeat(
                "hist_article_id",
                feature_max_idx["click_article_id"],
                embedding_dim,
                embedding_name="click_article_id",
            ),
            SEQ_LEN,
            "mean",
            "hist_len",
        ),
    ]
    item_feature_columns = [
        SparseFeat(
            "click_article_id", feature_max_idx["click_article_id"], embedding_dim
        )
    ]

    # 模型的定义
    # num_sampled: 负采样时的样本数量
    train_counter = Counter(train_model_input["click_article_id"])
    item_count = [
        train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)
    ]
    sampler_config = NegativeSampler(
        "frequency", num_sampled=5, item_name="click_article_id", item_count=item_count
    )

    import tensorflow as tf

    if tf.__version__ >= "2.0.0":
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)

    model = YoutubeDNN(
        user_feature_columns,
        item_feature_columns,
        user_dnn_hidden_units=(64, 16, embedding_dim),
        sampler_config=sampler_config,
    )
    # 模型编译
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)

    # 模型训练，这里可以定义验证集的比例，如果设置为0的话就是全量数据直接进行训练
    history = model.fit(
        train_model_input,
        train_label,
        batch_size=256,
        epochs=5,
        verbose=1,
        validation_split=0.0,
    )

    # 训练完模型之后,提取训练的Embedding，包括user端和item端
    test_user_model_input = test_model_input
    all_item_model_input = {"click_article_id": item_profile["click_article_id"].values}

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    # 保存当前的item_embedding 和 user_embedding 排序的时候可能能够用到，但是需要注意保存的时候需要和原始的id对应
    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    # embedding保存之前归一化一下
    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    # 将Embedding转换成字典的形式方便查询
    raw_user_id_emb_dict = {
        user_index_2_rawid[k]: v for k, v in zip(user_profile["user_id"], user_embs)
    }
    raw_item_id_emb_dict = {
        item_index_2_rawid[k]: v
        for k, v in zip(item_profile["click_article_id"], item_embs)
    }
    # 将Embedding保存到本地
    pickle.dump(raw_user_id_emb_dict, open(save_path + "/user_youtube_emb.pkl", "wb"))
    pickle.dump(raw_item_id_emb_dict, open(save_path + "/item_youtube_emb.pkl", "wb"))

    # faiss紧邻搜索，通过user_embedding 搜索与其相似性最高的topk个item
    index = faiss.IndexFlatIP(embedding_dim)
    # 上面已经进行了归一化，这里可以不进行归一化了
    #     faiss.normalize_L2(user_embs)
    #     faiss.normalize_L2(item_embs)
    index.add(item_embs)  # 将item向量构建索引
    sim, idx = index.search(
        np.ascontiguousarray(user_embs), topk
    )  # 通过user去查询最相似的topk个item

    user_recall_items_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(
            zip(test_user_model_input["user_id"], sim, idx)
    ):
        target_raw_id = user_index_2_rawid[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_index_2_rawid[rele_idx]
            user_recall_items_dict[target_raw_id][rele_raw_id] = (
                    user_recall_items_dict.get(target_raw_id, {}).get(rele_raw_id, 0)
                    + sim_value
            )

    user_recall_items_dict = {
        k: sorted(v.items(), key=lambda x: x[1], reverse=True)
        for k, v in user_recall_items_dict.items()
    }
    # 将召回的结果进行排序

    # 保存召回的结果
    # 这里是直接通过向量的方式得到了召回结果，相比于上面的召回方法，上面的只是得到了i2i及u2u的相似性矩阵，还需要进行协同过滤召回才能得到召回结果
    # 可以直接对这个召回结果进行评估，为了方便可以统一写一个评估函数对所有的召回结果进行评估
    pickle.dump(user_recall_items_dict, open(save_path + "/youtube_u2i_dict.pkl", "wb"))
    return user_recall_items_dict


# ----------------itemcf召回
# 基于商品的召回i2i
def item_based_recommend(
        user_id,
        user_item_time_dict,
        i2i_sim,
        sim_item_topk,
        recall_item_num,
        item_topk_click,
        item_created_time_dict,
        emb_i2i_sim,
):
    """
    基于文章协同过滤的召回
    :param user_id: 用户id
    :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
    :param i2i_sim: 字典，文章相似性矩阵
    :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
    :param recall_item_num: 整数， 最后的召回文章数量
    :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
    :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

    :return: 召回的文章列表 [(item1, score1), (item2, score2)...]
    """

    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}

    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[
                      :sim_item_topk
                      ]:
            if j in user_hist_items_:
                continue

            # 文章创建时间差权重
            created_time_weight = np.exp(
                0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j])
            )
            # 相似文章和历史点击文章序列中历史文章所在的位置权重
            loc_weight = 0.9 ** (len(user_hist_items) - loc)

            content_weight = 1.0
            if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                content_weight += emb_i2i_sim[i][j]
            if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                content_weight += emb_i2i_sim[j][i]

            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items():  # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = -i - 100  # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[
                :recall_item_num
                ]

    return item_rank


if __name__ == "__main__":
    data_path = "../data"
    save_path = "../multi_recall_result"
    # 做召回评估的一个标志, 如果不进行评估就是直接使用全量数据进行召回
    metric_recall = True

    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    # 采样数据
    # all_click_df = get_all_click_sample(data_path)

    # 全量训练集
    all_click_df = get_all_click_df(data_path, offline=False)

    # 对时间戳进行归一化,用于在关联规则的时候计算权重
    all_click_df["click_timestamp"] = all_click_df[["click_timestamp"]].apply(
        max_min_scaler
    )

    item_info_df = get_item_info_df(data_path)

    item_emb_dict = get_item_emb_dict(data_path)

    # 获取文章的属性信息，保存成字典的形式方便查询
    item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(
        item_info_df
    )

    # 定义一个多路召回字典，将各路召回的结果都保存在这个字典当中
    user_multi_recall_dict = {
        "itemcf_sim_itemcf_recall": {},
        "embedding_sim_item_recall": {},
        "youtubednn_recall": {},
        "youtubednn_usercf_recall": {},
        "cold_start_recall": {},
    }

    # 提取最后一次点击作为召回评估，如果不需要做召回评估直接使用全量的训练集进行召回(线下验证模型)
    # 如果不是召回评估，直接使用全量数据进行召回，不用将最后一次提取出来
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)

    # 创建文章相似度矩阵并储存
    # i2i_sim = itemcf_sim(all_click_df, item_created_time_dict)

    # 由于usercf计算时候太耗费内存了，这里就不直接运行了
    # 如果是采样的话，是可以运行的
    user_activate_degree_dict = get_user_activate_degree_dict(all_click_df)
    # u2u_sim = usercf_sim(all_click_df, user_activate_degree_dict)

    # 使用faiss根据embedding求相似度
    # item_emb_df = pd.read_csv(data_path + "/articles_emb.csv")
    # emb_i2i_sim = embdding_sim(
    #     all_click_df, item_emb_df, save_path, topk=10
    # )  # topk可以自行设置

    # 由于这里需要做召回评估，所以讲训练集中的最后一次点击都提取了出来
    """
    if not metric_recall:
        user_multi_recall_dict["youtubednn_recall"] = youtubednn_u2i_dict(
            all_click_df, topk=20
        )
    else:
        # trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
        user_multi_recall_dict["youtubednn_recall"] = youtubednn_u2i_dict(
            trn_hist_click_df, save_path, topk=20
        )
        # 召回效果评估
        metrics_recall(
            user_multi_recall_dict["youtubednn_recall"], trn_last_click_df, topk=20
        )
    """

    # ---------------itemcf召回---------------------
    # 先进行itemcf召回, 为了召回评估，所以提取最后一次点击

    if metric_recall:
        # 如果trn_hist_df, trn_last_click_df 为None
        if trn_hist_click_df is None and trn_last_click_df is None:
            trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    else:
        trn_hist_click_df = all_click_df

    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = get_user_item_time(trn_hist_click_df)

    i2i_sim = pickle.load(open(save_path + "/itemcf_i2i_sim.pkl", "rb"))
    emb_i2i_sim = pickle.load(open(save_path + "/emb_i2i_sim.pkl", "rb"))

    sim_item_topk = 20
    recall_item_num = 10
    item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

    # 对每个用户进行itemcf召回，其中考虑文章冷启动问题
    print("\nItemCF recall start")
    for user in tqdm(trn_hist_click_df["user_id"].unique()):
        user_recall_items_dict[user] = item_based_recommend(
            user,
            user_item_time_dict,
            i2i_sim,
            sim_item_topk,
            recall_item_num,
            item_topk_click,
            item_created_time_dict,
            emb_i2i_sim,
        )

    user_multi_recall_dict["itemcf_sim_itemcf_recall"] = user_recall_items_dict
    pickle.dump(
        user_multi_recall_dict["itemcf_sim_itemcf_recall"],
        open(save_path + "itemcf_recall_dict.pkl", "wb"),
    )

    if metric_recall:
        # 召回效果评估
        metrics_recall(
            user_multi_recall_dict["itemcf_sim_itemcf_recall"],
            trn_last_click_df,
            topk=recall_item_num,
        )

    # ---------------embedding召回---------------------
    # 这里是为了召回评估，所以提取最后一次点击
    """
    if metric_recall:
        trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    else:
        trn_hist_click_df = all_click_df

    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = get_user_item_time(trn_hist_click_df)
    i2i_sim = pickle.load(open(save_path + "emb_i2i_sim.pkl", "rb"))

    sim_item_topk = 20
    recall_item_num = 10

    item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

    for user in tqdm(trn_hist_click_df["user_id"].unique()):
        user_recall_items_dict[user] = item_based_recommend(
            user,
            user_item_time_dict,
            i2i_sim,
            sim_item_topk,
            recall_item_num,
            item_topk_click,
            item_created_time_dict,
            emb_i2i_sim,
        )

    user_multi_recall_dict["embedding_sim_item_recall"] = user_recall_items_dict
    pickle.dump(
        user_multi_recall_dict["embedding_sim_item_recall"],
        open(save_path + "embedding_sim_item_recall.pkl", "wb"),
    )

    if metric_recall:
        # 召回效果评估
        metrics_recall(
            user_multi_recall_dict["embedding_sim_item_recall"],
            trn_last_click_df,
            topk=recall_item_num,
        )
    """
