# import packages
import time, math, os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
import collections

warnings.filterwarnings("ignore")


# --------------------------------df节约内存函数
def reduce_mem(df):
    starttime = time.time()
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print(
        "-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min".format(
            end_mem,
            100 * (start_mem - end_mem) / start_mem,
            (time.time() - starttime) / 60,
        )
    )
    return df


# --------------------------------读取采样或全量数据函数
# debug模式：从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path, sample_nums=10000):
    """
    训练集中采样一部分数据调试

    :param data_path: 原数据的存储路径
    :param sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    :return: 采样之后的数据
    """
    all_click = pd.read_csv(data_path + "/train_click_log.csv")
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(
        all_user_ids, size=sample_nums, replace=False
    )  # 随机选择一部分做训练
    all_click = all_click[all_click["user_id"].isin(sample_user_ids)]

    # 删除数据中的重复项
    all_click = all_click.drop_duplicates(
        (["user_id", "click_article_id", "click_timestamp"])
    )
    return all_click


# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path, offline=True):
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


# --------------------------------获取 用户 - 文章 - 点击时间字典
# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):
    click_df = click_df.sort_values("click_timestamp")

    def make_item_time_pair(df):
        return list(zip(df["click_article_id"], df["click_timestamp"]))

    user_item_time_df = (
        click_df.groupby("user_id")[
            "click_article_id", "click_timestamp"
        ]  # 以用户id分组，获取用户点击的文章id和点击时间，返回的都是dataframe.groupby对象
        .apply(
            lambda x: make_item_time_pair(x)
        )  # 对分组对象进行操作，将用户点击的文章id和点击时间组成元组，然后转换为列表
        .reset_index()  # 重新设置索引，在第一列添加默认索引
        .rename(columns={0: "item_time_list"})  # 对列名进行重命名
    )
    # 将用户点击文章序列转换成字典的形式
    user_item_time_dict = dict(
        zip(user_item_time_df["user_id"], user_item_time_df["item_time_list"])
    )

    return user_item_time_dict


# --------------------------------获取点击最多的topk个文章
def get_item_topk_click(click_df, k):
    topk_click = click_df["click_article_id"].value_counts().index[:k].values
    return topk_click


# --------------------------------itemcf的物品相似度计算
def itemcf_sim(df):
    """
    计算文章与文章之间的相似性矩阵

    :param df: 数据表
    :item_created_time_dict: 文章创建时间的字典
    :return: 文章与文章的相似性矩阵
    """
    user_item_time_dict = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    print("构建物品相似度矩阵")
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += 1 / math.log(
                    1 + len(item_time_list)
                )  # 1/log(1+N),N为用户点击文章的数量,对异常活跃的用户进行一个惩罚

    i2i_sim_ = i2i_sim.copy()
    print("\n控制对热门物品的惩罚")
    for i, related_items in tqdm(i2i_sim.items()):
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(
                item_cnt[i] * item_cnt[j]
            )  # 考虑到了物品的热门程度，取alpha=0.5

    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + "/itemcf_i2i_sim.pkl", "wb"))
    return i2i_sim_


# 简单测试一下
"""
# 获取前10个点击最多的文章
topk_click = get_item_topk_click(all_click_df, k=10)
# 获取用户—文章—点击时间的字典
user_item_time_dict = get_user_item_time(all_click_df)
"""


# --------------------------------itemcf的文章推荐
# 基于商品的召回i2i
def item_based_recommend(
    user_id,
    user_item_time_dict,
    i2i_sim,
    sim_item_topk,
    recall_item_num,
    item_topk_click,
) -> dict:
    """
    基于文章协同过滤的召回

    :param user_id: 用户id
    :param user_item_time_dict: 字典, 根据点击时间获取的用户的点击文章序列 {user1: [(item1, time1), (item2, time2)..]...}
    :param i2i_sim: 字典，文章相似性矩阵 {item1: {item2: sim1, item3: sim2}, item2: {item1: sim1, item4: sim2}...
    :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
    :param recall_item_num: 整数， 最后的召回文章数量
    :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
    :return: 召回的文章列表字典 {item1: score1, item2: score2, ...}
    """
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {item_id for item_id, _ in user_hist_items}
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        # 根据文章i找到与文章i最相似的文章j，取前sim_item_topk个
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[
            :sim_item_topk
        ]:
            if j in user_hist_items_:
                continue
            item_rank.setdefault(j, 0)
            item_rank[j] += wij  # 如果有一篇文章和多篇历史文章相似，累加相似度

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items():  # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = -i - 100  # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break

    # 重新排序
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[
        :recall_item_num
    ]

    return item_rank


# --------------------------------生成提交文件
def submit(recall_df, save_path, topk=5, model_name=None):
    # 按照预测分数pred_score进行升序排序
    recall_df = recall_df.sort_values(by=["user_id", "pred_score"])
    # 增加一列文章排名，得分高的排名靠前
    recall_df["rank"] = recall_df.groupby(["user_id"])["pred_score"].rank(
        ascending=False, method="first"
    )

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby("user_id").apply(lambda x: x["rank"].max())
    assert tmp.min() >= topk

    del recall_df["pred_score"]
    submit = (
        recall_df[recall_df["rank"] <= topk]
        .set_index(["user_id", "rank"])
        .unstack(-1)
        .reset_index()
    )
    submit.columns = [
        int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)
    ]
    # 按照提交格式定义列名
    submit = submit.rename(
        columns={
            "": "user_id",
            1: "article_1",
            2: "article_2",
            3: "article_3",
            4: "article_4",
            5: "article_5",
        }
    )
    save_name = (
        save_path + model_name + "_" + datetime.today().strftime("%m-%d") + ".csv"
    )
    submit.to_csv(save_name, index=False, header=True)


if __name__ == "__main__":
    # 定义文件路径
    data_path = "../data"
    save_path = "../result"

    # 加载数据
    all_click_df = get_all_click_df(data_path, offline=False)

    # 求取物品相似度矩阵并保存
    i2i_sim = itemcf_sim(all_click_df)

    # 给每个用户根据物品的协同过滤推荐文章
    # 定义待训练的字典, 用户召回文章的列表, 用户id为key, 被推荐的文章字典为value
    user_recall_items_dict = collections.defaultdict(dict)

    # 获取 用户 - 文章 - 点击时间的字典
    user_item_time_dict = get_user_item_time(all_click_df)

    # 去取文章相似度
    i2i_sim = pickle.load(open(save_path + "/itemcf_i2i_sim.pkl", "rb"))

    # 相似文章的数量
    sim_item_topk = 10

    # 召回文章数量（需要推荐给用户的文章数量）
    recall_item_num = 10

    # 用户热度补全, 取前50个热门商品
    item_topk_click = get_item_topk_click(all_click_df, k=50)

    print("\nitemcf召回")
    for user in tqdm(all_click_df["user_id"].unique()):
        user_recall_items_dict[user] = item_based_recommend(
            user,
            user_item_time_dict,
            i2i_sim,
            sim_item_topk,
            recall_item_num,
            item_topk_click,
        )

    # --------------------------------召回字典转换成df
    # 将字典的形式转换成df
    user_item_score_list = []

    print("\n召回字典转换成df")
    for user, items in tqdm(user_recall_items_dict.items()):
        for item, score in items:
            user_item_score_list.append([user, item, score])

    recall_df = pd.DataFrame(
        user_item_score_list, columns=["user_id", "click_article_id", "pred_score"]
    )

    # 获取测试集
    test_click = pd.read_csv(data_path + "/testA_click_log.csv")
    test_users = test_click["user_id"].unique()

    # 从所有的召回数据中将测试集中的用户选出来
    test_recall = recall_df[recall_df["user_id"].isin(test_users)]

    # 生成测试提交数据
    submit(test_recall, save_path, topk=5, model_name="itemcf_baseline")
