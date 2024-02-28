from feature_engineering import *

data_path = "../data"
multi_recall_path = "../multi_recall_result"
save_path = "../feature_engineering_result"
click_trn, click_val, click_tst, val_ans = get_trn_val_tst_data(data_path, offline=True)

trn_user_item_feats_df = pd.read_csv(save_path + "/trn_user_item_feats_df.csv")
val_user_item_feats_df = pd.read_csv(save_path + "/val_user_item_feats_df.csv")
tst_user_item_feats_df = pd.read_csv(save_path + "/tst_user_item_feats_df.csv")

# 读取文章特征
articles = pd.read_csv(data_path + "/articles.csv")
articles = reduce_mem(articles)
# 日志数据，就是前面的所有数据
if click_val is not None:
    all_data = click_trn._append(click_val)
all_data = all_data._append(click_tst)

# 拼上文章信息
# merge函数的作用是将两个DataFrame进行合并，这里是根据click_article_id和article_id进行合并
# 其中若两组数据没有重复的列名，则将会将数据舍弃
all_data = all_data.merge(articles, left_on="click_article_id", right_on="article_id")

# # 用户活跃度特征和文章热度特征，值越小说明用户越活跃，文章越热门
# user_act_fea = active_level(all_data, ["user_id", "click_article_id", "click_timestamp"])
# article_hot_fea = hot_level(all_data, ["user_id", "click_article_id", "click_timestamp"])
#
# # 设备特征(这里时间会比较长)
# device_cols = [
#     "user_id",
#     "click_environment",
#     "click_deviceGroup",
#     "click_os",
#     "click_country",
#     "click_region",
#     "click_referrer_type",
# ]
# user_device_info = device_fea(all_data, device_cols)
#
# # 用户时间习惯特征
# user_time_hob_cols = ["user_id", "click_timestamp", "created_at_ts"]
# user_time_hob_info = user_time_hob_fea(all_data, user_time_hob_cols)
#
# # 用户主题爱好特征
# user_category_hob_cols = ["user_id", "category_id"]
# user_cat_hob_info = user_cat_hob_fea(all_data, user_category_hob_cols)
#
# # 用户的字数特征
# user_wcou_info = all_data.groupby("user_id")["words_count"].agg("mean").reset_index()
# user_wcou_info.rename(columns={"words_count": "words_hbo"}, inplace=True)
#
# # 所有表进行合并
# user_info = pd.merge(user_act_fea, user_device_info, on="user_id")
# user_info = user_info.merge(user_time_hob_info, on="user_id")
# user_info = user_info.merge(user_cat_hob_info, on="user_id")
# user_info = user_info.merge(user_wcou_info, on="user_id")
#
# # 保存特征
# user_info.to_csv(save_path + "/user_info.csv", index=False)

# ------------------------------用户特征直接读入
user_info = pd.read_csv(save_path + "/user_info.csv")
if os.path.exists(save_path + "/trn_user_item_feats_df.csv"):
    trn_user_item_feats_df = pd.read_csv(save_path + "/trn_user_item_feats_df.csv")

if os.path.exists(save_path + "/tst_user_item_feats_df.csv"):
    tst_user_item_feats_df = pd.read_csv(save_path + "/tst_user_item_feats_df.csv")

if os.path.exists(save_path + "/val_user_item_feats_df.csv"):
    val_user_item_feats_df = pd.read_csv(save_path + "/val_user_item_feats_df.csv")
else:
    val_user_item_feats_df = None

# 拼上用户特征
# 下面是线下验证的
trn_user_item_feats_df = trn_user_item_feats_df.merge(user_info, on="user_id", how="left")
if val_user_item_feats_df is not None:
    val_user_item_feats_df = val_user_item_feats_df.merge(user_info, on="user_id", how="left")
else:
    val_user_item_feats_df = None
tst_user_item_feats_df = tst_user_item_feats_df.merge(user_info, on="user_id", how="left")

print(trn_user_item_feats_df.columns)

# 拼上文章特征
trn_user_item_feats_df = trn_user_item_feats_df.merge(articles, left_on="click_article_id", right_on="article_id")
if val_user_item_feats_df is not None:
    val_user_item_feats_df = val_user_item_feats_df.merge(
        articles, left_on="click_article_id", right_on="article_id"
    )
else:
    val_user_item_feats_df = None
tst_user_item_feats_df = tst_user_item_feats_df.merge(articles, left_on="click_article_id", right_on="article_id")

# ------------------------------召回文章的主题是否在用户的爱好里面
trn_user_item_feats_df["is_cat_hab"] = trn_user_item_feats_df.apply(
    lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1
)
if val_user_item_feats_df is not None:
    val_user_item_feats_df["is_cat_hab"] = val_user_item_feats_df.apply(
        lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1
    )
else:
    val_user_item_feats_df = None
tst_user_item_feats_df["is_cat_hab"] = tst_user_item_feats_df.apply(
    lambda x: 1 if x.category_id in set(x.cate_list) else 0, axis=1
)

# 线下验证
del trn_user_item_feats_df["cate_list"]

if val_user_item_feats_df is not None:
    del val_user_item_feats_df["cate_list"]
else:
    val_user_item_feats_df = None

del tst_user_item_feats_df["cate_list"]

del trn_user_item_feats_df["article_id"]

if val_user_item_feats_df is not None:
    del val_user_item_feats_df["article_id"]
else:
    val_user_item_feats_df = None

del tst_user_item_feats_df["article_id"]

# 保存特征
# 训练验证特征
trn_user_item_feats_df.to_csv(save_path + "/trn_user_item_feats_df.csv", index=False)
if val_user_item_feats_df is not None:
    val_user_item_feats_df.to_csv(save_path + "/val_user_item_feats_df.csv", index=False)
tst_user_item_feats_df.to_csv(save_path + "/tst_user_item_feats_df.csv", index=False)
