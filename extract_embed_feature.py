import os
import torch
import torch.utils.data as data
from tqdm import tqdm
from tensorboard_logger import Logger
import time
import datetime
import numpy as np
import options
import utils
from dataset import dataset
from model_S import S_Model
from model_I import I_Model
from train import S_train, I_train
from test import S_test, I_test
from log import save_config, initial_log, save_best_record
from ranking import reliability_ranking


def main(args):
    # # >> Initialize the task
    # save_config(args, os.path.join(args.output_path_s1, "config.json"))
    # utils.set_seed(args.seed)
    # os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    # args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    #
    # model = S_Model(args)
    # model.load_state_dict(torch.load(os.path.join(args.model_path_s1, "model1_seed_{}.pkl".format(args.seed))))
    # model = model.to(args.device)
    # model.eval()
    # train_loader = data.DataLoader(dataset(args, phase="train", sample="random", stage=args.stage),
    #                                batch_size=1, shuffle=True, num_workers=args.num_workers)
    # test_loader = data.DataLoader(dataset(args, phase="test", sample="random", stage=args.stage),
    #                               batch_size=1, shuffle=False, num_workers=args.num_workers)
    # os.makedirs(f'/home/yunchuan/HR-Pro/dataset/{args.dataset}/features/embed_train',exist_ok=True)
    # os.makedirs(f'/home/yunchuan/HR-Pro/dataset/{args.dataset}/features/embed_test', exist_ok=True)
    # with torch.no_grad():
    #     for sample in train_loader:
    #         data_, vid_name = sample['data'], sample['vid_name']
    #         data_ = data_.to(args.device)
    #         output = model.extract_feature(data_)
    #         feature = output.cpu().numpy()
    #         np.save('/home/yunchuan/HR-Pro/dataset/{}/features/embed_train/{}.npy'.format(args.dataset,vid_name[0]),feature)
    #
    #     for sample in test_loader:
    #         data_, vid_name = sample['data'], sample['vid_name']
    #         data_ = data_.to(args.device)
    #         output = model.extract_feature(data_)
    #         feature = output.cpu().numpy()
    #         np.save('/home/yunchuan/HR-Pro/dataset/{}/features/embed_test/{}.npy'.format(args.dataset,vid_name[0]),feature)


    if 1:
        # # 定义文件夹路径
        origin_dir = '/home/yunchuan/HR-Pro/dataset/THUMOS14/features/origin_train'
        embed_dir = '/home/yunchuan/HR-Pro/dataset/THUMOS14/features/embed_train'
        fused_dir = '/home/yunchuan/HR-Pro/dataset/THUMOS14/features/fused_train'
        #
        # # 如果 fused_train 文件夹不存在，则创建
        # if not os.path.exists(fused_dir):
        #     os.makedirs(fused_dir)
        #
        # # 遍历 origin_train 文件夹下的所有文件（假设均为 .npy 文件）
        # for file_name in os.listdir(origin_dir):
        #     # 构造完整的文件路径
        #     origin_file = os.path.join(origin_dir, file_name)
        #     embed_file = os.path.join(embed_dir, file_name)
        #
        #     # 检查 embed_train 文件夹中是否存在对应的文件
        #     if not os.path.exists(embed_file):
        #         print(f"Warning: {embed_file} 文件不存在，跳过。")
        #         continue
        #
        #     # 加载两个数组
        #     arr_origin = np.load(origin_file)
        #     arr_embed = np.load(embed_file)
        #
        #     # 检查两个数组维度是否一致
        #     if arr_origin.shape[0] != arr_embed.shape[0]:
        #         print(f"Warning: 文件 {file_name} 两个数组在第一个维度上的长度不一致，跳过。")
        #         continue
        #
        #     # 在第二个维度上拼接数组，得到形状 (T, 2D)
        #     fused_arr = np.concatenate([arr_origin, arr_embed], axis=1)
        #
        #     # 保存拼接后的数组到 fused_train 文件夹下，文件名保持一致
        #     fused_file = os.path.join(fused_dir, file_name)
        #     np.save(fused_file, fused_arr)
        #
        #     print(f"已处理并保存: {file_name}")

        # 如果 fused_train 文件夹不存在，则创建
        if not os.path.exists(fused_dir):
            os.makedirs(fused_dir)

        # 遍历 origin_train 文件夹下的所有文件（假设均为 .npy 文件）
        for file_name in os.listdir(origin_dir):
            origin_file = os.path.join(origin_dir, file_name)
            embed_file = os.path.join(embed_dir, file_name)

            # 检查 embed_train 文件夹中是否存在对应的文件
            if not os.path.exists(embed_file):
                print(f"Warning: {embed_file} 文件不存在，跳过。")
                continue

            # 加载两个数组
            arr_origin = np.load(origin_file)
            arr_embed = np.load(embed_file)

            # 检查两个数组是否形状一致
            if arr_origin.shape != arr_embed.shape:
                print(f"Warning: 文件 {file_name} 的两个数组形状不同，跳过。")
                continue

            # 计算元素级平均值，结果形状依然为 (T, D)
            # fused_arr = (arr_origin + arr_embed) / 2.0
            fused_arr = (arr_origin + arr_embed * 0.1)

            # 保存结果到 fused_train 文件夹下，文件名保持一致
            fused_file = os.path.join(fused_dir, file_name)
            np.save(fused_file, fused_arr)

            print(f"已处理并保存: {file_name}")

        # # 定义文件夹路径
        origin_dir = '/home/yunchuan/HR-Pro/dataset/THUMOS14/features/origin_test'
        embed_dir = '/home/yunchuan/HR-Pro/dataset/THUMOS14/features/embed_test'
        fused_dir = '/home/yunchuan/HR-Pro/dataset/THUMOS14/features/fused_test'
        #
        # # 如果 fused_train 文件夹不存在，则创建
        # if not os.path.exists(fused_dir):
        #     os.makedirs(fused_dir)
        #
        # # 遍历 origin_train 文件夹下的所有文件（假设均为 .npy 文件）
        # for file_name in os.listdir(origin_dir):
        #     # 构造完整的文件路径
        #     origin_file = os.path.join(origin_dir, file_name)
        #     embed_file = os.path.join(embed_dir, file_name)
        #
        #     # 检查 embed_train 文件夹中是否存在对应的文件
        #     if not os.path.exists(embed_file):
        #         print(f"Warning: {embed_file} 文件不存在，跳过。")
        #         continue
        #
        #     # 加载两个数组
        #     arr_origin = np.load(origin_file)
        #     arr_embed = np.load(embed_file)
        #
        #     # 检查两个数组维度是否一致
        #     if arr_origin.shape[0] != arr_embed.shape[0]:
        #         print(f"Warning: 文件 {file_name} 两个数组在第一个维度上的长度不一致，跳过。")
        #         continue
        #
        #     # 在第二个维度上拼接数组，得到形状 (T, 2D)
        #     fused_arr = np.concatenate([arr_origin, arr_embed], axis=1)
        #
        #     # 保存拼接后的数组到 fused_train 文件夹下，文件名保持一致
        #     fused_file = os.path.join(fused_dir, file_name)
        #     np.save(fused_file, fused_arr)
        #
        #     print(f"已处理并保存: {file_name}")

        # 如果 fused_train 文件夹不存在，则创建
        if not os.path.exists(fused_dir):
            os.makedirs(fused_dir)

        # 遍历 origin_train 文件夹下的所有文件（假设均为 .npy 文件）
        for file_name in os.listdir(origin_dir):
            origin_file = os.path.join(origin_dir, file_name)
            embed_file = os.path.join(embed_dir, file_name)

            # 检查 embed_train 文件夹中是否存在对应的文件
            if not os.path.exists(embed_file):
                print(f"Warning: {embed_file} 文件不存在，跳过。")
                continue

            # 加载两个数组
            arr_origin = np.load(origin_file)
            arr_embed = np.load(embed_file)

            # 检查两个数组是否形状一致
            if arr_origin.shape != arr_embed.shape:
                print(f"Warning: 文件 {file_name} 的两个数组形状不同，跳过。")
                continue

            # 计算元素级平均值，结果形状依然为 (T, D)
            # fused_arr = (arr_origin + arr_embed) / 2.0
            fused_arr = (arr_origin + arr_embed * 0.1)

            # 保存结果到 fused_train 文件夹下，文件名保持一致
            fused_file = os.path.join(fused_dir, file_name)
            np.save(fused_file, fused_arr)

            print(f"已处理并保存: {file_name}")




if __name__ == "__main__":
    # np.load('/home/yunchuan/HR-Pro/dataset/THUMOS14/features/test_fuse/video_test_0001558.npy')
    args = options.parse_args()
    main(args)