import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import logging
import time
import lightgbm as lgb
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")


# -------------------------df节省内存函数
# 节省内存的一个函数
# 减少内存
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


# ---------------------------划分训练集和验证集
# all_click_df指的是训练集
# sample_user_nums 训练集中采样作为验证集的用户数量
def trn_val_spilt(all_click_df, sample_user_nums):
    all_click = all_click_df
    all_user_ids = all_click.user_id.unique()

    # replace=True表示可以重复抽样，反之不可以
    sample_user_ids = np.random.choice(
        all_user_ids, size=sample_user_nums, replace=False
    )

    click_val = all_click[all_click["user_id"].isin(sample_user_ids)]
    click_trn = all_click[~all_click["user_id"].isin(sample_user_ids)]

    # 将验证集中的最后一次点击给抽取出来作为答案
    click_val = click_val.sort_values(by=["user_id", "click_timestamp"])
    val_ans = click_val.groupby("user_id").tail(1)

    # 去除val_ans中某些用户只有一个点击数据的情况，如果该用户只有一个点击数据，又被分到ans中，
    # 那么训练集中就没有这个用户的点击数据，出现用户冷启动问题，给自己模型验证带来麻烦
    click_val = (
        click_val.groupby("user_id").apply(lambda x: x[:-1]).reset_index(drop=True)
    )

    val_ans = val_ans[
        val_ans.user_id.isin(click_val.user_id.unique())
    ]  # 保证答案中出现的用户再验证集中还有
    click_val = click_val[click_val.user_id.isin(val_ans.user_id.unique())]

    return click_trn, click_val, val_ans


# 获取当前历史点击和最后一次点击
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


# 获取训练、验证、测试和测试集
def get_trn_val_tst_data(data_path, offline=True, sample_user_num=10000):
    if offline:
        click_trn_data = pd.read_csv(data_path + "/train_click_log.csv")
        click_trn_data = reduce_mem(click_trn_data)
        click_trn, click_val, val_ans = trn_val_spilt(click_trn_data, sample_user_num)
    else:
        click_trn = pd.read_csv(data_path + "/train_click_log.csv")
        click_trn = reduce_mem(click_trn)
        click_val = None
        val_ans = None

    click_tst = pd.read_csv(data_path + "/testA_click_log.csv")

    return click_trn, click_val, click_tst, val_ans


# 读取召回列表
# 返回多路召回列表或者单路召回
def get_recall_list(save_path, single_recall_model=None, multi_recall=False):
    if multi_recall:
        return pickle.load(open(save_path + "final_recall_items_dict.pkl", "rb"))

    if single_recall_model == "i2i_itemcf":
        return pickle.load(open(save_path + "itemcf_recall_dict.pkl", "rb"))
    elif single_recall_model == "i2i_emb_itemcf":
        return pickle.load(open(save_path + "itemcf_emb_dict.pkl", "rb"))
    elif single_recall_model == "user_cf":
        return pickle.load(open(save_path + "youtubednn_usercf_dict.pkl", "rb"))
    elif single_recall_model == "youtubednn":
        return pickle.load(open(save_path + "youtube_u2i_dict.pkl", "rb"))


# ------------------------------各种embedding
def train_item_word2vec(
    click_df, embed_size=64, sace_name="item_w2v_emb.pkl", spilt_char=" "
):
    click_df = click_df.sort_values("click_timestamp")
    # 只有转换成字符串才可以进行训练
    click_df["click_article_id"] = click_df["click_article_id"].astype(str)
    # 转换成句子的形式
    docs = (
        click_df.groupby(["user_id"])["click_article_id"]
        .apply(lambda x: list(x))
        .reset_index()
    )
    docs = docs["click_article_id"].values.tolist()

    # 为了方便查看训练的进度，这里设定一个log信息
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO
    )

    # 这里的参数对训练得到的向量影响也很大,默认负采样为5
    w2v = Word2Vec(
        docs, size=16, sg=1, window=5, seed=2020, workers=24, min_count=1, iter=1
    )

    # 保存成字典的形式
    item_w2v_emb_dict = {k: w2v[k] for k in click_df["click_article_id"]}
    pickle.dump(item_w2v_emb_dict, open(save_path + "item_w2v_emb.pkl", "wb"))

    return item_w2v_emb_dict


# 可以通过字典查询对应的item的Embedding
def get_embedding(save_path, all_click_df):
    if os.path.exists(save_path + "item_content_emb.pkl"):
        item_content_emb_dict = pickle.load(
            open(save_path + "item_content_emb.pkl", "rb")
        )
    else:
        print("item_content_emb.pkl 文件不存在...")

    # w2v Embedding是需要提前训练好的
    if os.path.exists(save_path + "item_w2v_emb.pkl"):
        item_w2v_emb_dict = pickle.load(open(save_path + "item_w2v_emb.pkl", "rb"))
    else:
        item_w2v_emb_dict = train_item_word2vec(all_click_df)

    if os.path.exists(save_path + "item_youtube_emb.pkl"):
        item_youtube_emb_dict = pickle.load(
            open(save_path + "item_youtube_emb.pkl", "rb")
        )
    else:
        print("item_youtube_emb.pkl 文件不存在...")

    if os.path.exists(save_path + "user_youtube_emb.pkl"):
        user_youtube_emb_dict = pickle.load(
            open(save_path + "user_youtube_emb.pkl", "rb")
        )
    else:
        print("user_youtube_emb.pkl 文件不存在...")

    return (
        item_content_emb_dict,
        item_w2v_emb_dict,
        item_youtube_emb_dict,
        user_youtube_emb_dict,
    )


# 读取文章信息
def get_article_info_df():
    article_info_df = pd.read_csv(data_path + "articles.csv")
    article_info_df = reduce_mem(article_info_df)

    return article_info_df


# -----------------------读取数据
data_path = "../data"
save_path = "../result"
click_trn, click_val, click_tst, val_ans = get_trn_val_tst_data(data_path, offline=True)

click_trn_hist, click_trn_last = get_hist_and_last_click(click_trn)
if click_val is not None:
    click_val_hist, click_val_last = click_val, val_ans
else:
    click_val_hist, click_val_last = None, None

click_tst_hist = click_tst


# -------------------------对训练数据做负采样
# 将召回列表转换成df的形式
def recall_dict_2_df(recall_list_dict):
    df_row_list = []  # [user, item, score]
    for user, recall_list in tqdm(recall_list_dict.items()):
        for item, score in recall_list:
            df_row_list.append([user, item, score])

    col_names = ["user_id", "sim_item", "score"]
    recall_list_df = pd.DataFrame(df_row_list, columns=col_names)

    return recall_list_df


# 负采样函数，这里可以控制负采样时的比例, 这里给了一个默认的值
def neg_sample_recall_data(recall_items_df, sample_rate=0.001):
    pos_data = recall_items_df[recall_items_df["label"] == 1]
    neg_data = recall_items_df[recall_items_df["label"] == 0]

    print(
        "pos_data_num:",
        len(pos_data),
        "neg_data_num:",
        len(neg_data),
        "pos/neg:",
        len(pos_data) / len(neg_data),
    )

    # 分组采样函数
    def neg_sample_func(group_df):
        neg_num = len(group_df)
        sample_num = max(int(neg_num * sample_rate), 1)  # 保证最少有一个
        sample_num = min(
            sample_num, 5
        )  # 保证最多不超过5个，这里可以根据实际情况进行选择
        return group_df.sample(n=sample_num, replace=True)

    # 对用户进行负采样，保证所有用户都在采样后的数据中
    neg_data_user_sample = neg_data.groupby("user_id", group_keys=False).apply(
        neg_sample_func
    )
    # 对文章进行负采样，保证所有文章都在采样后的数据中
    neg_data_item_sample = neg_data.groupby("sim_item", group_keys=False).apply(
        neg_sample_func
    )

    # 将上述两种情况下的采样数据合并
    neg_data_new = neg_data_user_sample.append(neg_data_item_sample)
    # 由于上述两个操作是分开的，可能将两个相同的数据给重复选择了，所以需要对合并后的数据进行去重
    neg_data_new = neg_data_new.sort_values(["user_id", "score"]).drop_duplicates(
        ["user_id", "sim_item"], keep="last"
    )

    # 将正样本数据合并
    data_new = pd.concat([pos_data, neg_data_new], ignore_index=True)

    return data_new


# 召回数据打标签
def get_rank_label_df(recall_list_df, label_df, is_test=False):
    # 测试集是没有标签了，为了后面代码同一一些，这里直接给一个负数替代
    if is_test:
        recall_list_df["label"] = -1
        return recall_list_df

    label_df = label_df.rename(columns={"click_article_id": "sim_item"})
    recall_list_df_ = recall_list_df.merge(
        label_df[["user_id", "sim_item", "click_timestamp"]],
        how="left",
        on=["user_id", "sim_item"],
    )
    recall_list_df_["label"] = recall_list_df_["click_timestamp"].apply(
        lambda x: 0.0 if np.isnan(x) else 1.0
    )
    del recall_list_df_["click_timestamp"]

    return recall_list_df_


def get_user_recall_item_label_df(
    click_trn_hist,
    click_val_hist,
    click_tst_hist,
    click_trn_last,
    click_val_last,
    recall_list_df,
):
    # 获取训练数据的召回列表
    trn_user_items_df = recall_list_df[
        recall_list_df["user_id"].isin(click_trn_hist["user_id"].unique())
    ]
    # 训练数据打标签
    trn_user_item_label_df = get_rank_label_df(
        trn_user_items_df, click_trn_last, is_test=False
    )
    # 训练数据负采样
    trn_user_item_label_df = neg_sample_recall_data(trn_user_item_label_df)

    if click_val is not None:
        val_user_items_df = recall_list_df[
            recall_list_df["user_id"].isin(click_val_hist["user_id"].unique())
        ]
        val_user_item_label_df = get_rank_label_df(
            val_user_items_df, click_val_last, is_test=False
        )
        val_user_item_label_df = neg_sample_recall_data(val_user_item_label_df)
    else:
        val_user_item_label_df = None

    # 测试数据不需要进行负采样，直接对所有的召回商品进行打-1标签
    tst_user_items_df = recall_list_df[
        recall_list_df["user_id"].isin(click_tst_hist["user_id"].unique())
    ]
    tst_user_item_label_df = get_rank_label_df(tst_user_items_df, None, is_test=True)

    return trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df


# 读取召回列表
recall_list_dict = get_recall_list(
    save_path, single_recall_model="i2i_itemcf"
)  # 这里只选择了单路召回的结果，也可以选择多路召回结果
# 将召回数据转换成df
recall_list_df = recall_dict_2_df(recall_list_dict)
