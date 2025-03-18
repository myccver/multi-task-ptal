import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import utils
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



class FramePrediction(nn.Module):
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim * 4, feature_dim)  # 输入维度为 4*F

    def forward(self, combined_features):
        """输入为拼接后的左右特征 [N, 4*F]"""
        x = F.relu(self.fc1(combined_features))
        return x  # 输出 [N, F]


# 定义顺序预测网络
class SequenceOrderPrediction(nn.Module):
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim * 5, feature_dim)
        # self.dropout  = nn.Dropout(0.5)
        self.fc2 = nn.Linear(feature_dim, 1)

    def forward(self, feature_sequence):
        # 输入: [batch_size, 3*feature_dim]
        # 输出: [batch_size, 1]
        x = F.relu(self.fc1(feature_sequence))
        x = torch.sigmoid(self.fc2(x))
        return x

# 重建网络

class FrameAutoencoder(nn.Module):
    def __init__(self, feat_dim=2048, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, latent_dim),
            #nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            # nn.Linear(latent_dim, latent_dim),
            #nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(latent_dim, feat_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

# 定义顺序扰动预测网络
class SequenceShufflePrediction(nn.Module):
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim * 5, feature_dim)
        # self.dropout  = nn.Dropout(0.5)
        self.fc2 = nn.Linear(feature_dim, 1)

    def forward(self, feature_sequence):
        # 输入: [batch_size, 3*feature_dim]
        # 输出: [batch_size, 1]
        x = F.relu(self.fc1(feature_sequence))
        x = torch.sigmoid(self.fc2(x))
        return x




class Reliable_Memory(nn.Module):
    def __init__(self, num_class, feat_dim):
        super(Reliable_Memory, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.proto_momentum = 0.001  # 0.001
        self.proto_num = 1
        self.proto_vectors = torch.nn.Parameter(torch.zeros([self.num_class, self.proto_num, self.feat_dim]), requires_grad=False)
        
    def init(self, args, net, train_loader):
        print('Memory initialization in progress...')
        with torch.no_grad():
            net.eval()
            pfeat_total = {}
            temp_loader = data.DataLoader(train_loader.dataset, batch_size=1, shuffle=False, num_workers=4)
            for sample in temp_loader:
                _data, vid_label, point_anno = sample['data'], sample['vid_label'], sample['point_label']
                # print(sample['vid_name'],sample['vid_len'])
                # if sample['vid_name']==['v_y47RXYfefvQ']:
                #     print(123)
                outputs = net(_data.to(args.device), vid_label.to(args.device))
                embeded_feature = outputs['embeded_feature']
                for b in range(point_anno.shape[0]):
                    gt_class = torch.nonzero(vid_label[b]).squeeze(1).numpy()
                    for c in gt_class:
                        select_id = torch.nonzero(point_anno[b, :, c]).squeeze(1)
                        if select_id.shape[0] > 0:
                            act_feat = embeded_feature[b, select_id, :]
                            if c not in pfeat_total.keys():
                                pfeat_total[c] = act_feat
                            else:
                                pfeat_total[c] = torch.cat([pfeat_total[c], act_feat])

            for c in range(self.num_class):
                cluster_centers = pfeat_total[c].mean(dim=0, keepdim=True)
                self.proto_vectors[c] = cluster_centers


    def update(self, args, feats, act_seq, vid_label):
        self.proto_vectors = self.proto_vectors.to(args.device)
        feat_list = {}
        for b in range(act_seq.shape[0]):
            gt_class = torch.nonzero(vid_label[b]).cpu().squeeze(1).numpy()
            for c in gt_class:
                select_id = torch.nonzero(act_seq[b, :, c]).squeeze(1)
                if select_id.shape[0] > 0:
                    act_feat = feats[b, select_id, :]
                    if c not in feat_list.keys():
                        feat_list[c] = act_feat
                    else:
                        feat_list[c] = torch.cat(feat_list[c], act_feat)

        for c in feat_list.keys():
            if len(feat_list[c]) > 0:
                feat_update = feat_list[c].mean(dim=0, keepdim=True)
                self.proto_vectors[c] = (1 - self.proto_momentum) * self.proto_vectors[c] + self.proto_momentum * feat_update


class Reliabilty_Aware_Block(nn.Module):
    def __init__(self, input_dim, dropout, num_heads=8, dim_feedforward=128, pos_embed=False):
        super(Reliabilty_Aware_Block, self).__init__()
        self.conv_query = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv_key = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv_value = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)

        self.self_atten = nn.MultiheadAttention(input_dim, num_heads=num_heads, dropout=0.1) # 0.1
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features, attn_mask=None,):
        src = features.permute(2, 0, 1)
        q = k = src
        q = self.conv_query(features).permute(2, 0, 1)
        k = self.conv_key(features).permute(2, 0, 1)

        src2, attn = self.self_atten(q, k, src, attn_mask=attn_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.permute(1, 2, 0)
        return src, attn


# class Encoder(nn.Module):
#     def __init__(self, args):
#         super(Encoder, self).__init__()
#         self.dataset = args.dataset
#         self.feature_dim = args.feature_dim
#
#         self.backbone1 = nn.Sequential(
#             nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             # nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
#             # nn.ReLU(),
#             # nn.Dropout(0.5),
#         )
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.feature_dim,
#             nhead=8,
#             dim_feedforward=128,
#             dropout=0.1,
#             batch_first=True
#         )
#
#         # 创建Transformer Encoder
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=2
#         )
#         # self.backbone2 = nn.Sequential(
#         #     nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.5),
#         # )
#         # self.backbone3 = nn.Sequential(
#         #     nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.5),
#         # )
#
#
#         self.feature_embedding = nn.Sequential(
#             nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#         )
#
#     def forward(self, input_features, prototypes=None):
#         '''
#         input_feature: [B,T,F]
#         prototypes：[C,1,F]
#         '''
#         B, T, F = input_features.shape
#         input_features = input_features.permute(0, 2, 1)                        #[B,F,T]
#         # prototypes = prototypes.to(input_features.device)                       #[C,1,F]
#         # prototypes = prototypes.view(1,F,-1).expand(B,-1,-1)                    #[B,F,C]
#         # #
#         # mixed_feature =  torch.cat([input_features, prototypes], dim=2)     #[B,F,T+C]
#         #
#         # if hasattr(self, 'RAB'):
#         #     layer_features = torch.cat([input_features, prototypes], dim=2)     #[B,F,T+C]
#         #     # layer_features = input_features  # [B,F,T]
#         #     for layer in self.RAB:
#         #         layer_features, _ = layer(layer_features)
#         #     input_features = layer_features[:, :, :T]                           #[B,F,T]
#         input_features = self.backbone1(input_features)
#         input_features = self.transformer_encoder(input_features.transpose(-1,-2)).transpose(-1,-2)
#         # input_features = self.backbone2(input_features) + input_features
#         # input_features = self.backbone3(input_features) + input_features
#         # input_features = input_features + self.backbone2(input_features)
#         embeded_features = self.feature_embedding(input_features)               #[B,F,T]
#
#         return embeded_features

  
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim

        RAB_args = args.RAB_args
        self.RAB = nn.ModuleList([
            Reliabilty_Aware_Block(
                input_dim=self.feature_dim,
                dropout=RAB_args['drop_out'],
                num_heads=RAB_args['num_heads'],
                dim_feedforward=RAB_args['dim_feedforward'])
            for i in range(RAB_args['layer_num'])
        ])

        self.feature_embedding = nn.Sequential(
            # nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )

    def forward(self, input_features, prototypes=None):
        '''
        input_feature: [B,T,F]
        prototypes：[C,1,F]
        '''
        B, T, F = input_features.shape
        input_features = input_features.permute(0, 2, 1)                        #[B,F,T]
        prototypes = prototypes.to(input_features.device)                       #[C,1,F]
        prototypes = prototypes.view(1,F,-1).expand(B,-1,-1)                    #[B,F,C]
        if hasattr(self, 'RAB'):
            layer_features = torch.cat([input_features, prototypes], dim=2)     #[B,F,T+C]
            # layer_features = input_features  # [B,F,T]
            for layer in self.RAB:
                layer_features, _ = layer(layer_features)
            input_features = layer_features[:, :, :T]                           #[B,F,T]
        embeded_features = self.feature_embedding(input_features)               #[B,F,T]

        return embeded_features
    

class S_Model(nn.Module):
    def __init__(self, args):
        super(S_Model, self).__init__()
        self.feature_dim = args.feature_dim
        self.num_class = args.num_class
        self.r_act = args.r_act
        self.dropout = args.dropout

        self.memory = Reliable_Memory(self.num_class, self.feature_dim)
        self.encoder = Encoder(args)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.num_class + 1, kernel_size=1, stride=1, padding=0, bias=False)
            )
        self.sigmoid = nn.Sigmoid()
        self.bce_criterion = nn.BCELoss(reduction='none')
        self.lambdas = args.lambdas

        self.args = args
        if self.args.mtl:
            args.feature_dim = args.feature_dim
            self.frame_predictor = FramePrediction(args.feature_dim)
            self.sequenceorder_predictor = SequenceOrderPrediction(args.feature_dim)
            # self.frame_autoencoder = FrameAutoencoder(args.feature_dim, latent_dim=args.feature_dim)
            self.shuffle_predictor = SequenceShufflePrediction(args.feature_dim)
            self.adapter = nn.Sequential(nn.Linear(args.feature_dim , args.feature_dim),
                                         nn.ReLU(),
                                         # nn.Dropout(0.5),
                                         nn.Linear(args.feature_dim, args.feature_dim),
                                         # nn.ReLU(),
                                         # # nn.Dropout(0.5),
                                         # nn.Linear(args.feature_dim, args.feature_dim),
                                         )  # 更深的特征变换

            # 门控网络
            self.gate = nn.Sequential(
                nn.Linear(args.feature_dim, 3),
                nn.ReLU(),
                # nn.Linear(128, 3),
                nn.Softmax(dim=-1)
            )
        

    def extract_feature(self, input_features):
        embeded_feature = self.encoder(input_features, self.memory.proto_vectors)
        adapted_features = self.adapter(embeded_feature.transpose(-1, -2)).transpose(-1, -2)
        # return embeded_feature.squeeze(0).transpose(0,1)
        return adapted_features.squeeze(0).transpose(0, 1)

    def forward(self, input_features, vid_labels=None, point_labels=None):
        '''
        input_feature: [B,T,F]
        '''
        # >> Encoder and classifier
        embeded_feature = self.encoder(input_features, self.memory.proto_vectors)   #[B,F,T]
        # embeded_feature = embeded_feature + input_features.transpose(-1,-2)
        cas = self.classifier(embeded_feature)                                      #[B,C+1,T]
        cas = cas.permute(0, 2, 1)                                                  #[B,T,C+1]
        cas = self.sigmoid(cas)                                                     #[B,T,C+1]
        # class-Specific activation sequence
        cas_S = cas[:, :, :-1]                                                      #[B,T,C]
        # class-Agnostic attention sequence (background)
        bkg_score = cas[:, :, -1]                                                   #[B,T]

        # >> Fusion
        cas_P = cas_S * (1 - bkg_score.unsqueeze(2))                                #[B,T,C]
        cas_fuse = torch.cat((cas_P, bkg_score.unsqueeze(2)), dim=2)                #[B,T,C+1]

        # >> Top-k pooling
        value, _ = cas_S.sort(descending=True, dim=1)
        k_act = max(1, input_features.shape[1] // self.r_act)
        topk_scores = value[:, :k_act, :]
        if vid_labels is None:
            vid_score = torch.mean(topk_scores, dim=1)
        else:
            vid_score = (torch.mean(topk_scores, dim=1) * vid_labels) + \
                        (torch.mean(cas_S, dim=1) * (1 - vid_labels))
        #--------------------
        # if self.args.mtl:
        #     if point_labels is None:
        #         predicted_tensors = None
        #         target_tensors = None
        #     else:
        #         predicted_list = []
        #         target_list = []
        #
        #         for i in range(2, embeded_feature.shape[2] - 2):  # 遍历序列（忽略第 0 和 最后一个）
        #             if point_labels[0, i, :].sum() >= 1:  # 如果是标注点
        #                 left_feature = embeded_feature[:, :, i - 2:i].transpose(1, 2).reshape(1, -1)  # 左侧特征 [1, 2048]
        #                 # right_feature = embeded_feature[:, :, i + 1:i + 3].transpose(1, 2).reshape(1,
        #                 #                                                                            -1)  # 右侧特征 [1, 2048]
        #                 right_feature = embeded_feature[:, :, i+1:i+3].flip(dims=[2]).transpose(1, 2).reshape(1, -1)  # 右侧特征 [1, 2048]
        #                 target_feature = embeded_feature[:, :, i]  # 目标特征 [1, 2048]
        #
        #                 predicted_feature = self.frame_predictor(left_feature, right_feature)  # 预测特征 [1, 2048]
        #
        #                 predicted_list.append(predicted_feature)
        #                 target_list.append(target_feature)
        #         predicted_tensors = torch.cat(predicted_list, dim=0)
        #         target_tensors = torch.cat(target_list, dim=0)
        #     ##
        #     import random
        #
        #     if point_labels is None:
        #         sequence_predictions = None
        #         sequence_labels = None
        #     else:
        #         ordered_predictions = []  # 存储新的预测特征
        #         ordered_labels = []  # 存储新的二分类标签
        #
        #         for i in range(1, embeded_feature.shape[2] - 1):  # 遍历序列，忽略第 0 和最后一个
        #             if point_labels[0, i, :].sum() >= 1:  # 如果是标注点
        #                 # 提取前一帧、当前帧（标注帧）、后一帧
        #                 previous_feature = embeded_feature[:, :, i - 1].reshape(1, -1)  # 前一帧特征
        #                 middle_feature = embeded_feature[:, :, i].reshape(1, -1)  # 中间帧（标注帧）
        #                 next_feature = embeded_feature[:, :, i + 1].reshape(1, -1)  # 后一帧特征
        #
        #                 # 随机选择顺序，可能是123或321
        #                 frame_order = random.choice([(previous_feature, middle_feature, next_feature),
        #                                              (next_feature, middle_feature, previous_feature)])
        #
        #                 # 将选定的顺序拼接
        #                 combined_features = torch.cat(frame_order, dim=0)  # 拼接特征，维度[3, 2048]
        #
        #                 # 使用 SequenceOrderPrediction 进行预测
        #                 sequence_prediction = self.sequenceorder_predictor(combined_features)  # 输入整个序列
        #
        #                 # 生成标签
        #                 if torch.equal(frame_order[0], previous_feature) and \
        #                         torch.equal(frame_order[1], middle_feature) and \
        #                         torch.equal(frame_order[2], next_feature):
        #                     label = 1  # 顺序正确
        #                 else:
        #                     label = 0  # 顺序反转
        #
        #                 # 保存预测结果和标签
        #                 ordered_predictions.append(sequence_prediction)
        #                 ordered_labels.append(label)
        #
        #         # 拼接所有预测特征和标签
        #         sequence_predictions = torch.cat(ordered_predictions, dim=0)
        #         sequence_labels = torch.tensor(ordered_labels) .to(sequence_predictions.device) # 转换为tensor
        #
        #     return dict(
        #         cas_fuse=cas_fuse,  # [B,T,C+1]
        #         cas_S=cas_S,  # [B,T,C+1]
        #         vid_score=vid_score,  # [B,C]
        #         embeded_feature=embeded_feature.permute(0, 2, 1),  # [B,T,F]
        #         predicted_tensors=predicted_tensors,
        #         target_tensors=target_tensors,
        #         sequence_predictions = sequence_predictions,
        #         sequence_labels = sequence_labels,
        #
        #     )
        if self.args.mtl:



            adapted_features = self.adapter(embeded_feature.transpose(-1,-2)).transpose(-1,-2)
            B, F, T = adapted_features.shape  # 输入形状 [1, 2048, T]
            device = adapted_features.device
            gate_weights = self.gate(adapted_features.transpose(-1,-2)).transpose(-1,-2)

            frame_predict_weight = gate_weights[:, 0, :].unsqueeze(1)  # [1,1,265]
            sequence_weight = gate_weights[:, 1, :].unsqueeze(1)  # [B, 1, L]
            # autoencoder_weight = gate_weights[:, 2, :].unsqueeze(1)
            shuffle_weight = gate_weights[:, 2, :].unsqueeze(1)

            frame_predict_feature = frame_predict_weight * adapted_features
            sequence_feature = sequence_weight * adapted_features
            # autoencoder_feature = autoencoder_weight * adapted_features
            shuffle_feature = shuffle_weight * adapted_features


            # ===== 初始化所有返回值 =====
            predicted_tensors = None
            target_tensors = None
            sequence_predictions = None
            sequence_labels = None
            labeled_features = None
            reconstructions = None
            shuffle_predictions = None
            shuffle_labels = None


            # ================= 特征预测任务（维度修正） =================
            if point_labels is not None:
                # 左窗口：从i-2开始，窗口大小2
                # left_windows = embeded_feature.unfold(2, 2, 1)  # [1, F, T-1, 2]
                left_windows = frame_predict_feature.unfold(2, 2, 1)  # [1, F, T-1, 2]

                # 右窗口：从i+1开始，窗口大小2，翻转
                # 注意：直接生成所有可能的右窗口（无需偏移）
                # right_windows = embeded_feature.unfold(2, 2, 1).flip(dims=[3])  # [1, F, T-1, 2]
                right_windows = frame_predict_feature.unfold(2, 2, 1).flip(dims=[3])  # [1, F, T-1, 2]

                # 有效索引处理（确保i-2和i+1不越界）
                valid_mask = (point_labels[0, 2:T - 2, :].sum(dim=1) >= 1)  # i ∈ [2, T-3]
                valid_indices = valid_mask.nonzero().squeeze(dim=1) + 2  # 调整到原数据索引

                if valid_indices.numel() > 0:
                    # 提取左右窗口特征
                    left_features = left_windows[:, :, valid_indices - 2, :]  # [1, F, N, 2]
                    right_features = right_windows[:, :, valid_indices + 1, :]  # [1, F, N, 2]

                    # 合并特征并调整维度（关键修改点）
                    combined = torch.cat([left_features, right_features], dim=3)  # [1, F, N, 4]
                    combined = combined.permute(0, 2, 3, 1).reshape(-1, 4 * F)  # [N, 4 * 2048]

                    # 直接传入拼接后的特征
                    predicted_tensors = self.frame_predictor(combined)  # [N, 2048]
                    # target_tensors = embeded_feature[0, :, valid_indices].t()  # [N, 2048]
                    target_tensors = frame_predict_feature[0, :, valid_indices].t()  # [N, 2048]
                    # target_tensors = input_features.transpose(-1,-2)[0, :, valid_indices].t()  # [N, 2048]

            # ================= 序列顺序预测（维度修正） =================
            # if point_labels is not None:
            #     # triplet_windows = embeded_feature.unfold(2, 3, 1)  # [1, F, T-2, 3]
            #     triplet_windows = sequence_feature.unfold(2, 3, 1)  # [1, F, T-2, 3]
            #     valid_mask = (point_labels[0, 1:T - 1, :].sum(dim=1) >= 1)
            #     valid_windows = triplet_windows[:, :, valid_mask, :]  # [1, F, N, 3]
            #
            #     if valid_windows.numel() > 0:
            #         # 保持批量维度
            #         reverse_mask = torch.rand(valid_windows.size(2), device=device) < 0.5  # [N]
            #         reversed_windows = valid_windows.clone()
            #         reversed_windows[:, :, reverse_mask, :] = reversed_windows[:, :, reverse_mask, :].flip(dims=[3])
            #
            #         # 调整维度（支持批量输入）
            #         combined = reversed_windows.permute(0, 2, 3, 1).reshape(-1, 3 * F)  # [N, 3 * 2048]
            #         sequence_predictions = self.sequenceorder_predictor(combined)  # [N, 1]
            #         sequence_labels = (~reverse_mask).float().to(device)  # [N]

            # if point_labels is not None:
            #     # 生成三元组窗口 [batch, features, num_windows, window_size]
            #     # triplet_windows = embeded_feature.unfold(2, 3, 1)  # [1, F, T-2, 3]
            #     triplet_windows = sequence_feature.unfold(2, 3, 1)  # [1, F, T-2, 3]
            #
            #     # 选择有效中心帧 (对应窗口索引范围验证)
            #     valid_center_mask = (point_labels[0, 1:T - 1, :].sum(dim=1) >= 1)  # [T-2]
            #     valid_windows = triplet_windows[:, :, valid_center_mask, :]  # [1, F, N, 3]
            #
            #     if valid_windows.numel() > 0:
            #         # 生成正序和反序配对数据
            #         original_windows = valid_windows
            #         reversed_windows = valid_windows.flip(dims=[-1])  # 反转最后一个维度（窗口顺序）
            #
            #         # 合并正反样本 [1, F, 2*N, 3]
            #         combined_windows = torch.cat([original_windows, reversed_windows], dim=2)
            #
            #         # 调整维度结构 [2*N, 3*F]
            #         batch_size, num_features, num_windows, window_size = combined_windows.shape
            #         combined = combined_windows.permute(0, 2, 3, 1).reshape(-1, window_size * num_features)
            #
            #         # 生成配对标签 (前N个为1表示正序，后N个为0表示反序)
            #         sequence_labels = torch.cat([
            #             torch.ones(num_windows // 2, device=device),  # 原顺序标签为1
            #             torch.zeros(num_windows // 2, device=device)  # 反序标签为0
            #         ]).float()
            #
            #         # 进行预测
            #         sequence_predictions = self.sequenceorder_predictor(combined)  # [2*N, 1]

            if point_labels is not None:
                # 生成五元组窗口 [batch, features, num_windows, window_size]
                quintuplet_windows = sequence_feature.unfold(2, 5, 1)  # [1, F, T-4, 5]

                # 选择有效中心帧 (窗口中心索引为2，对应原始序列的2:T-2)
                valid_center_mask = (point_labels[0, 2:T - 2, :].sum(dim=1) >= 1)  # [T-4]
                valid_windows = quintuplet_windows[:, :, valid_center_mask, :]  # [1, F, N, 5]

                if valid_windows.numel() > 0:
                    # 生成正序和反序配对数据
                    original_windows = valid_windows
                    reversed_windows = valid_windows.flip(dims=[-1])  # 反转窗口顺序

                    # 合并正反样本 [1, F, 2*N, 5]
                    combined_windows = torch.cat([original_windows, reversed_windows], dim=2)

                    # 调整维度结构 [2*N, 5*F]
                    batch_size, num_features, num_windows, window_size = combined_windows.shape
                    combined = combined_windows.permute(0, 2, 3, 1).reshape(-1, window_size * num_features)

                    # 生成配对标签 (直接使用原始有效窗口数N)
                    N = valid_windows.size(2)  # 从valid_windows直接获取
                    sequence_labels = torch.cat([
                        torch.ones(N, device=device),  # 原顺序标签为1
                        torch.zeros(N, device=device)  # 反序标签为0
                    ]).float()

                    # 进行预测
                    sequence_predictions = self.sequenceorder_predictor(combined)  # [2*N, 1]

            # #------------------重建-------------------------------
            # if point_labels is not None:
            #     # 提取被标注的帧特征 [batch, features, time]
            #     # 假设 point_labels 形状为 [batch, time, num_classes]
            #     labeled_mask = (point_labels.sum(dim=-1) > 0)  # [batch, time]
            #
            #     # 获取有效特征 (仅处理有标注的帧)
            #     # labeled_features = embeded_feature[:, :, labeled_mask[0]]  # [1, F, N]
            #     labeled_features = autoencoder_feature[:, :, labeled_mask[0]]  # [1, F, N]
            #
            #     if labeled_features.size(2) > 0:
            #         # 转置为 [N, F] 作为自编码器输入
            #         labeled_features = labeled_features.permute(2, 1, 0).squeeze(-1)  # [N, F]
            #
            #         # 自编码重建（输入=输出）
            #         reconstructions = self.frame_autoencoder(labeled_features)  # [N, F]

             #--------------------乱序--------------------
            if point_labels is not None:
                # 生成五元组窗口 [batch, features, num_windows, window_size]
                quintuplet_windows = shuffle_feature.unfold(2, 5, 1)  # [1, F, T-4, 5]

                # 选择有效中心帧 (中间帧索引为2)
                valid_center_mask = (point_labels[0, 2:T - 2, :].sum(dim=1) >= 1)  # [T-4]
                valid_windows = quintuplet_windows[:, :, valid_center_mask, :]  # [1, F, N, 5]

                if valid_windows.numel() > 0:
                    # 正序样本：保持原窗口 [1, F, N, 5]
                    original_windows = valid_windows

                    # 乱序样本：中间帧固定，周围四帧随机打乱
                    shuffled_windows = []
                    for i in range(valid_windows.size(2)):
                        # 保持四维结构 [1, F, 1, 5]
                        window = valid_windows[:, :, i:i + 1, :]  # 关键修改：i:i+1 保留维度

                        # 提取周围四帧（索引0,1,3,4）并打乱顺序
                        surrounding_indices = torch.tensor([0, 1, 3, 4], device=device)
                        shuffled_idx = torch.randperm(4)  # 生成随机排列
                        new_surrounding = window[:, :, :, surrounding_indices[shuffled_idx]]  # [1, F, 1, 4]

                        # 重组窗口：打乱的周围四帧 + 中间帧 [1, F, 1, 5]
                        shuffled_window = torch.cat([
                            new_surrounding[:, :, :, :2],  # 前两帧
                            window[:, :, :, 2:3],  # 中间帧（保持原位）
                            new_surrounding[:, :, :, 2:]  # 后两帧
                        ], dim=3)  # 关键修改：dim=3（时间步维度）

                        shuffled_windows.append(shuffled_window)

                    # 合并乱序样本 [1, F, N, 5]
                    shuffled_windows = torch.cat(shuffled_windows, dim=2)  # dim=2 合并窗口数量

                    # 合并正序和乱序样本 [1, F, 2N, 5]
                    combined_windows = torch.cat([original_windows, shuffled_windows], dim=2)

                    # 调整维度结构 [2N, 5*F]
                    batch_size, num_features, num_windows, window_size = combined_windows.shape
                    combined = combined_windows.permute(0, 2, 3, 1).reshape(-1, window_size * num_features)

                    # 生成标签：前N为1（正序），后N为0（乱序）
                    N = valid_windows.size(2)
                    shuffle_labels = torch.cat([
                        torch.ones(N, device=device),  # 正序标签1
                        torch.zeros(N, device=device)  # 乱序标签0
                    ]).float()

                    # 进行预测
                    shuffle_predictions = self.shuffle_predictor(combined)  # [2N, 1]



            return {
                "cas_fuse": cas_fuse,
                "cas_S": cas_S,
                "vid_score": vid_score,
                "embeded_feature": embeded_feature.permute(0, 2, 1),
                "predicted_tensors": predicted_tensors,
                "target_tensors": target_tensors,
                "sequence_predictions": sequence_predictions,
                "sequence_labels": sequence_labels,
                # 'labeled_features': labeled_features,
                # 'reconstructions': reconstructions,
                'shuffle_predictions': shuffle_predictions,
                'shuffle_labels': shuffle_labels,

            }
        else:
            return dict(
                cas_fuse=cas_fuse,  # [B,T,C+1]
                cas_S=cas_S,  # [B,T,C+1]
                vid_score=vid_score,  # [B,C]
                embeded_feature=embeded_feature.permute(0, 2, 1),  # [B,T,F]
            )



    def criterion(self, args, outputs, vid_label, point_label):
        vid_score, embeded_feature, cas_fuse = outputs['vid_score'], outputs['embeded_feature'], outputs['cas_fuse']
        point_label = torch.cat((point_label, torch.zeros((point_label.shape[0], point_label.shape[1], 1)).to(args.device)), dim=2)
        act_seed, bkg_seed = utils.select_seed(cas_fuse[:, :, -1].detach().cpu(), point_label.detach().cpu())

        loss_dict = {}
        if not self.args.mtl:
            # >> base loss
            loss_vid, loss_frame, loss_frame_bkg = self.base_loss_func(args, act_seed, bkg_seed, vid_score, vid_label,
                                                                       cas_fuse, point_label)
            loss_dict["loss_vid"] = loss_vid
            loss_dict["loss_frame"] = loss_frame
            loss_dict["loss_frame_bkg"] = loss_frame_bkg

            # >> feat loss
            loss_contrastive = self.feat_loss_func(args, embeded_feature, act_seed, bkg_seed, vid_label)
            loss_dict["loss_contrastive"] = loss_contrastive


        # # triplet loss
        # loss_triplet = self.triplet_loss(args, embeded_feature.squeeze(0), point_label, bkg_seed.to(point_label.device), margin=0.2,
        #                                  norm_feat=True, hard_mining=True)
        # loss_dict["loss_triplet"] = loss_triplet

        # predict_loss
        if self.args.mtl:
            # >> base loss
            loss_vid, loss_frame, loss_frame_bkg = self.base_loss_func(args, act_seed, bkg_seed, vid_score, vid_label,
                                                                       cas_fuse, point_label)
            loss_dict["loss_vid"] = loss_vid
            loss_dict["loss_frame"] = loss_frame
            loss_dict["loss_frame_bkg"] = loss_frame_bkg

            # >> feat loss
            loss_contrastive = self.feat_loss_func(args, embeded_feature, act_seed, bkg_seed, vid_label)
            loss_dict["loss_contrastive"] = loss_contrastive
            loss_predict = self.prediction_loss(args, outputs['predicted_tensors'], outputs['target_tensors'])
            loss_dict["loss_predict"] = loss_predict
            loss_sequence_order = self.sequence_order_loss(args,outputs['sequence_predictions'], outputs['sequence_labels'])
            loss_dict['loss_sequence_order'] = loss_sequence_order
            # loss_reconstructions = self.reconstruction_loss(args,outputs['labeled_features'], outputs['reconstructions'])
            # loss_dict['loss_reconstructions'] = loss_reconstructions
            loss_shuffle = self.sequence_shuffle_loss(args, outputs['shuffle_predictions'], outputs['shuffle_labels'])
            loss_dict['loss_shuffle'] = loss_shuffle

        # >> update memory
        self.memory.update(args, embeded_feature.detach(), act_seed, vid_label)

        if self.args.mtl:
            loss_total = self.lambdas[0] * loss_vid + self.lambdas[1] * loss_frame \
                         + self.lambdas[2] * loss_frame_bkg + self.lambdas[
                             3] * loss_contrastive+ 0.5 * loss_predict + 0.5 * loss_sequence_order  + 0.5 *  loss_shuffle #+ 0 * loss_reconstructions # + 0.5 * loss_triplet
        else:
            loss_total = self.lambdas[0] * loss_vid + self.lambdas[1] * loss_frame \
                         + self.lambdas[2] * loss_frame_bkg + self.lambdas[
                             3] * loss_contrastive

        loss_dict["loss_total"] = loss_total

        return loss_total, loss_dict


    def base_loss_func(self, args, act_seed, bkg_seed, vid_score, vid_label, cas_sigmoid_fuse, point_anno):
        # >> video-level loss
        loss_vid = self.bce_criterion(vid_score, vid_label)
        loss_vid = loss_vid.mean()

        # >> frame-level loss
        loss_frame = 0
        loss_frame_bkg = 0
        # act frame loss
        act_seed = act_seed.to(args.device)
        focal_weight_act = (1 - cas_sigmoid_fuse ) * point_anno + cas_sigmoid_fuse * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2
        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1)
        loss_frame = (((focal_weight_act * self.bce_criterion(cas_sigmoid_fuse, point_anno) * weighting_seq_act)
                        .sum(dim=2)).sum(dim=1) / (num_actions + 1e-6)).mean()
        # # 使用act_seed
        # focal_weight_act = (1 - cas_sigmoid_fuse) * act_seed + cas_sigmoid_fuse * (1 - act_seed)
        # focal_weight_act = focal_weight_act ** 2
        # weighting_seq_act = act_seed.max(dim=2, keepdim=True)[0]
        # num_actions = act_seed.max(dim=2)[0].sum(dim=1)
        # loss_frame = (((focal_weight_act * self.bce_criterion(cas_sigmoid_fuse, act_seed) * weighting_seq_act)
        #                .sum(dim=2)).sum(dim=1) / (num_actions + 1e-6)).mean()
        # bkg frame loss
        bkg_seed = bkg_seed.unsqueeze(-1).to(args.device)
        point_anno_bkg = torch.zeros_like(point_anno).to(args.device)
        point_anno_bkg[:, :, -1] = 1
        weighting_seq_bkg = bkg_seed
        num_bkg = bkg_seed.sum(dim=1).squeeze(1)
        focal_weight_bkg = (1 - cas_sigmoid_fuse) * point_anno_bkg + cas_sigmoid_fuse * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2
        loss_frame_bkg = (((focal_weight_bkg * self.bce_criterion(cas_sigmoid_fuse, point_anno_bkg) * weighting_seq_bkg)
                            .sum(dim=2)).sum(dim=1) / (num_bkg + 1e-6)).mean()

        return loss_vid, loss_frame, loss_frame_bkg
    
    def feat_loss_func(self, args, embeded_feature, act_seed, bkg_seed, vid_label):
        if args.dataset == 'ActivityNet1.3':
            temperature = 0.1  # 对比学习温度系数
        else:
            temperature = 0.1  # 对比学习温度系数
        loss_contra = 0
        proto_vectors = utils.norm(self.memory.proto_vectors.to(args.device))                                        #[C,N,F]                                                             
        for b in range(act_seed.shape[0]):
            # >> extract pseudo-action/background features
            gt_class = torch.nonzero(vid_label[b]).squeeze(1)
            act_feat_lst = []
            for c in gt_class:
                act_feat_lst.append(utils.extract_region_feat(act_seed[b, :, c], embeded_feature[b, :, :]))
            bkg_feat = utils.extract_region_feat(bkg_seed[b].squeeze(-1), embeded_feature[b, :, :])
            if bkg_feat is None:
                print("Warning: bkg_feat is None, skipping...")
                bkg_feat = []
                # >> caculate similarity matrix
            if len(bkg_feat) == 0:
                continue
            bkg_feat = utils.norm(torch.cat(bkg_feat, 0))                                                            #[t_b,F]
            b_sim_matrix = torch.matmul(bkg_feat.unsqueeze(0).expand(args.num_class, -1, -1), 
                                        torch.transpose(proto_vectors, 1, 2)) / temperature                                 #[C,t_b,N]
            b_sim_matrix = torch.exp(b_sim_matrix).reshape(b_sim_matrix.shape[0], -1).mean(dim=-1)                   #[C]
            for idx, act_feat in enumerate(act_feat_lst):
                if act_feat is not None:
                    if len(act_feat) == 0:
                        continue
                    act_feat = utils.norm(torch.cat(act_feat, 0))                                                    #[t_a,F]
                    a_sim_matrix = torch.matmul(act_feat.unsqueeze(0).expand(args.num_class, -1, -1), 
                                                torch.transpose(proto_vectors, 1, 2)) / temperature                          #[C,t_a,N]
                    a_sim_matrix = torch.exp(a_sim_matrix).reshape(a_sim_matrix.shape[0], -1).mean(dim=-1)           #[C]                                                      

            # >> caculate contrastive loss
                    c = gt_class[idx]
                    loss_contra_act = - torch.log(a_sim_matrix[c] / a_sim_matrix.sum())
                    loss_contra_bkg = - torch.log(a_sim_matrix[c] / 
                                                 (a_sim_matrix[c] + b_sim_matrix[c]))
                    loss_contra += (0.5 * loss_contra_act + 0.5 * loss_contra_bkg)

            loss_contra = loss_contra / gt_class.shape[0]
        loss_contra = loss_contra / act_seed.shape[0]

        return loss_contra


    def triplet_loss(self, args, embedding, act_seed, bkg_seed, margin, norm_feat, hard_mining):
        r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
        Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
        Loss for Person Re-Identification'."""
        def transform_labels(tensor):
            """
            将标签矩阵转换为新的标签。具体规则如下：
            1. 如果某行中只有一个位置为1，新的标签为该位置的索引。
            2. 如果某行中所有位置都是0，新的标签为-1。
            3. 如果某行中有多个位置为1，新的标签为这些位置索引 + 1 后的乘积。

            参数：
            tensor (torch.Tensor): 输入的标签矩阵，shape 为 (num_samples, num_labels)。

            返回：
            torch.Tensor: 转换后的标签，shape 为 (num_samples,)。
            """
            # 获取样本数量
            num_samples = tensor.shape[0]

            # 创建一个新的张量，用来存储每个样本的新的标签，初始值为 0
            new_labels = torch.zeros(num_samples, dtype=torch.int)

            # 遍历每一个样本
            for i in range(num_samples):
                # 获取当前样本的标签行
                row = tensor[i]

                # 获取标签值为1的索引位置
                indices = row.nonzero(as_tuple=True)[0]

                # 如果该行没有1（即全为0），则新标签为 -1
                if len(indices) == 0:
                    new_labels[i] = -1
                # 如果该行只有一个1，新的标签为该1所在位置的索引
                elif len(indices) == 1:
                    new_labels[i] = indices[0]  # 新标签为索引位置
                else:
                    # 如果该行有多个1，新的标签是这些索引加1后的乘积
                    product = 1
                    for idx in indices:
                        product *= (idx + 1)  # 索引从0开始，所以需要加1
                    new_labels[i] = product  # 赋值为乘积结果

            return new_labels

        def softmax_weights(dist, mask):
            max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
            diff = dist - max_v
            Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
            W = torch.exp(diff) * mask / Z
            return W

        def hard_example_mining(dist_mat, is_pos, is_neg):
            """For each anchor, find the hardest positive and negative sample.
            Args:
              dist_mat: pair wise distance between samples, shape [N, M]
              is_pos: positive index with shape [N, M]
              is_neg: negative index with shape [N, M]
            Returns:
              dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
              dist_an: pytorch Variable, distance(anchor, negative); shape [N]
              p_inds: pytorch LongTensor, with shape [N];
                indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
              n_inds: pytorch LongTensor, with shape [N];
                indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
            NOTE: Only consider the case in which all labels have same num of samples,
              thus we can cope with all anchors in parallel.
            """

            assert len(dist_mat.size()) == 2

            # 避免报错
            if len(dist_mat * is_pos)==0 or len(dist_mat * is_neg + is_pos * 1e9)==0:
                return None, None
            # 避免报错

            # `dist_ap` means distance(anchor, positive)
            # both `dist_ap` and `relative_p_inds` with shape [N]
            dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
            # `dist_an` means distance(anchor, negative)
            # both `dist_an` and `relative_n_inds` with shape [N]
            dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

            return dist_ap, dist_an

        def weighted_example_mining(dist_mat, is_pos, is_neg):
            """For each anchor, find the weighted positive and negative sample.
            Args:
              dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
              is_pos:
              is_neg:
            Returns:
              dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
              dist_an: pytorch Variable, distance(anchor, negative); shape [N]
            """
            assert len(dist_mat.size()) == 2

            is_pos = is_pos
            is_neg = is_neg
            dist_ap = dist_mat * is_pos
            dist_an = dist_mat * is_neg

            weights_ap = softmax_weights(dist_ap, is_pos)
            weights_an = softmax_weights(-dist_an, is_neg)

            dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
            dist_an = torch.sum(dist_an * weights_an, dim=1)

            return dist_ap, dist_an


        def euclidean_dist(x, y):
            m, n = x.size(0), y.size(0)
            xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy - 2 * torch.matmul(x, y.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
            return dist

        def cosine_dist(x, y):
            x = F.normalize(x, dim=1)
            y = F.normalize(y, dim=1)
            dist = 2 - 2 * torch.mm(x, y.t())
            return dist
        # 预处理
        targets = transform_labels(torch.cat((act_seed[:,:,:-1],bkg_seed.unsqueeze(-1)),dim=-1).squeeze(0)).to(embedding.device)
        # 找出标签不为 -1 的样本索引
        valid_indices = targets != -1  # 返回一个布尔型 tensor，表示哪些标签不为 -1

        # 根据 valid_indices 筛选出对应的特征和标签
        embedding = embedding[valid_indices]  # 筛选出特征
        targets = targets[valid_indices]  # 筛选出标签

        if norm_feat:
            dist_mat = cosine_dist(embedding, embedding)
        else:
            dist_mat = euclidean_dist(embedding, embedding)

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        if hard_mining:
            dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
        else:
            dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)
        # 避免报错
        if dist_ap is None or dist_an is None:
            return torch.tensor(0.0).to(embedding.device)
        #

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            # fmt: off
            if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
            # fmt: on

        return loss

    # 定义损失函数
    def prediction_loss(self, args, predicted, target):
        # return F.mse_loss(predicted, target)  # L2 预测损失
        cos_sim = F.cosine_similarity(predicted, target, dim=-1)
        return 1 - cos_sim.mean()  # 余弦相似度越接近 1，损失应越小

    def sequence_order_loss(self,args, predicted, target):
        """
        顺序分类的损失函数，用于二分类任务。
        `predicted`：模型输出的预测概率（经过 sigmoid 的输出）。
        `target`：真实标签（1 表示顺序正确，0 表示反转顺序）。
        """
        # 确保 target 的形状与 predicted 一致
        target = target.view(-1, 1).float()  # 转换为 [N, 1] 的形状
        # 使用二元交叉熵损失（BCELoss）
        return F.binary_cross_entropy(predicted, target, reduction='mean')  # 计算交叉熵损失

    def reconstruction_loss(self,args, predicted, target):
        """基于L1范数的重建损失
        Args:
            predicted: 预测特征 [..., feature_dim]
            target:    目标特征 [..., feature_dim]
        Returns:
            loss: L1损失标量
        """
        # return F.mse_loss(predicted, target, reduction='mean')
        return 1 - F.cosine_similarity(predicted, target, dim=1).mean()

    def sequence_shuffle_loss(self,args, predicted, target):
        """
        顺序分类的损失函数，用于二分类任务。
        `predicted`：模型输出的预测概率（经过 sigmoid 的输出）。
        `target`：真实标签（1 表示顺序正确，0 表示反转顺序）。
        """
        # 确保 target 的形状与 predicted 一致
        target = target.view(-1, 1).float()  # 转换为 [N, 1] 的形状
        # 使用二元交叉熵损失（BCELoss）
        return F.binary_cross_entropy(predicted, target, reduction='mean')  # 计算交叉熵损失






