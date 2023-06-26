import torch
import torch.nn as nn
import copy

import numpy as np

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.W_q, self.W_k, self.W_v = nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim)

    def forward(self, src, prior, input_key_padding_mask):
        query, key, value = self.W_q(src + prior), self.W_k(src + prior), self.W_v(src)
        local_src, local_attention_weights = self.self_attn(query, key, value, key_padding_mask=input_key_padding_mask)

        src = src + self.dropout1(local_src)
        src = self.norm1(src)
        local_src = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(local_src)
        src = self.norm2(src)

        return src, local_attention_weights


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.W_q, self.W_k, self.W_v = nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim)

    def forward(self, src, prior, input_key_padding_mask, position_embed):
        query, key, value = self.W_q(src + prior), self.W_k(src + prior), self.W_v(src)
        global_src, global_attention_weights = self.self_attn(query=query+position_embed, key=key+position_embed,
            value=value, key_padding_mask=input_key_padding_mask)

        src = src + self.dropout2(global_src)
        src = self.norm3(src)
        global_src = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout3(global_src)

        return src, global_attention_weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, input, prior, input_key_padding_mask):
        output = input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, local_attention_weights = layer(output, prior, input_key_padding_mask)
            weights[i] = local_attention_weights
        if self.num_layers > 0:
            return output, weights
        else:
            return output, None


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, embed_dim):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, input, prior, input_key_padding_mask, position_embed):

        output = input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, global_attention_weights = layer(output, prior, input_key_padding_mask, position_embed)
            weights[i] = global_attention_weights

        if self.num_layers>0:
            return output, weights
        else:
            return output, None


class spatial_encoder(nn.Module):

    def __init__(self, enc_layer_num=1, embed_dim=1936, nhead=8, dim_feedforward=2048, dropout=0.1,
                 trainPrior=None, use_spatial_prior=False, obj_class_num=37, attention_class_num=3, spatial_class_num=6, contact_class_num=17):

        super(spatial_encoder, self).__init__()

        encoder_layer = TransformerEncoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, enc_layer_num)

        self.use_spatial_prior = use_spatial_prior

        self.initialPrior = torch.zeros((obj_class_num, 
                                         attention_class_num + spatial_class_num + contact_class_num))
       
        for idx, prior in trainPrior.items():
            obj_cls_idx = int(idx)
            self.initialPrior[obj_cls_idx] = torch.tensor(prior['initial_prior'])

        self.initialPrior = nn.Parameter(self.initialPrior)

        self.state_fc = nn.Sequential(nn.Linear(attention_class_num + spatial_class_num + contact_class_num, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 1936),)

    def forward(self, features, im_idx, entry, mode):

        rel_idx = torch.arange(im_idx.shape[0])
        bbox_num = torch.sum(im_idx == torch.mode(im_idx)[0]) # torch.mode->(tensor, longTensor), 返回众数以及众数下标
        frame_num = int(im_idx[-1] + 1)

        rel_input = torch.zeros([bbox_num, frame_num, features.shape[1]]).to(features.device)  # (bbox_num, frame_num, feature_dim)
        rel_prior = torch.zeros([bbox_num, frame_num, features.shape[1]]).to(features.device)  # (bbox_num, frame_num, feature_dim)

        priors = torch.zeros_like(features).to(features.device)                                # (bbox_num, feature_dim)
        masks = torch.zeros([frame_num, bbox_num], dtype=torch.uint8).to(features.device)      # (frame_num, bbox_num)

        # with prior knowledge
        for rel_idx in range(features.size(0)):
            bbox_idx = entry['pair_idx'][rel_idx, 1]
            label = int(entry['labels'][bbox_idx].item()) if mode == 'predcls' else int(entry['pred_labels'][bbox_idx].item())
            priors[rel_idx] = self.state_fc(self.initialPrior[label])

        for frame_idx in range(frame_num):
            rel_input[:torch.sum(im_idx == frame_idx), frame_idx, :] = features[im_idx == frame_idx]
            masks[frame_idx, torch.sum(im_idx == frame_idx):] = 1

            if self.use_spatial_prior:
                rel_prior[:torch.sum(im_idx == frame_idx), frame_idx, :] = priors[im_idx == frame_idx]

        rel_output, attention_weights = self.encoder(rel_input, rel_prior, masks)

        spatial_features = (rel_output.permute(1, 0, 2)).contiguous().view(-1, features.shape[1])[masks.view(-1) == 0] # (rel_num, feature_dim)
        spatial_priors = (rel_prior.permute(1, 0, 2)).contiguous().view(-1, features.shape[1])[masks.view(-1) == 0] # (rel_num, feature_dim)

        return spatial_features, spatial_priors


class temporal_decoder(nn.Module):

    def __init__(self, dec_layer_num=3, embed_dim=1936, nhead=8, dim_feedforward=2048, dropout=0.1, pred_contact_threshold=0.5,
                 trainPrior=None, use_temporal_prior=False, obj_class_num=37, attention_class_num=3, spatial_class_num=6, contact_class_num=17):
        
        super(temporal_decoder, self).__init__()

        decoder_layer = TransformerDecoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer, dec_layer_num, embed_dim)

        self.position_embedding = nn.Embedding(2, embed_dim)
        nn.init.uniform_(self.position_embedding.weight)

        self.use_temporal_prior = use_temporal_prior

        self.pred_contact_threshold = pred_contact_threshold

        self.initialPrior = torch.zeros((obj_class_num, 
                                         attention_class_num + spatial_class_num + contact_class_num))
        self.specificPrior = torch.zeros((obj_class_num, 
                                          contact_class_num, 
                                          attention_class_num + spatial_class_num + contact_class_num))

        for idx, prior in trainPrior.items():
            obj_cls_idx = int(idx)
            self.initialPrior[obj_cls_idx] = torch.tensor(prior['initial_prior'])

            for c_rel_idx, c_rel_prior in zip(prior['c_rel_idx'], prior['specific_prior']):
                self.specificPrior[obj_cls_idx][c_rel_idx] = torch.tensor(c_rel_prior)

        self.initialPrior, self.specificPrior = nn.Parameter(self.initialPrior), nn.Parameter(self.specificPrior)

        self.state_fc = nn.Sequential(nn.Linear(attention_class_num + spatial_class_num + contact_class_num, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 1936),)

        #self.temporal_context = nn.Linear(window_size * embed_dim, embed_dim)

    def forward(self, features, contact_distribution, im_idx, entry, mode):

        _contact_distribution = contact_distribution.detach().cpu().numpy()

        bbox_num = torch.sum(im_idx == torch.mode(im_idx)[0]) # torch.mode->(tensor, longTensor), 返回众数以及众数下标
        frame_num = int(im_idx[-1] + 1)

        history_anno_dict, history_score_dict = {}, {}
        temporal_priors, initialPrior, specificPrior = torch.zeros_like(features).to(features.device), torch.zeros_like(features).to(features.device), torch.zeros_like(features).to(features.device)

        for rel_idx in range(features.size(0)):

            bbox_idx = entry['pair_idx'][rel_idx, 1]
            bbox, label = entry['boxes'][bbox_idx], (int(entry['labels'][bbox_idx].item()) if mode == 'predcls' else int(entry['pred_labels'][bbox_idx].item()))
            obj_score = entry['pred_scores'][bbox_idx] if (mode != 'predcls') and (not self.training) else 1
            frame_idx = int(entry['im_idx'][rel_idx].item())

            if label not in history_anno_dict.keys():
                history_anno_dict[label] = [None for _ in range(frame_num)]
                history_score_dict[label] = torch.zeros([frame_num]).to(features.device)

            if self.training:
                history_score_dict[label][frame_idx] = 1.0
                
                if self.pred_contact_threshold == 1.0:
                    history_anno_dict[label][frame_idx] = set(entry['contacting_gt'][rel_idx])
                else:
                    history_anno_dict[label][frame_idx] = {8} if np.argmax(_contact_distribution[rel_idx]) == 8 else \
                                                          set(np.where(_contact_distribution[rel_idx] >= self.pred_contact_threshold)[0].tolist()) - {8}
            else:
                if mode == 'predcls':
                    history_score_dict[label][frame_idx] = 1.0

                    if self.pred_contact_threshold == 1.0:
                        history_anno_dict[label][frame_idx] = set(entry['contacting_gt'][rel_idx])
                    else:
                        history_anno_dict[label][frame_idx] = {8} if np.argmax(_contact_distribution[rel_idx]) == 8 else \
                                                              set(np.where(_contact_distribution[rel_idx] >= self.pred_contact_threshold)[0].tolist()) - {8}
                else:
                    if obj_score > history_score_dict[label][frame_idx]:
                        history_score_dict[label][frame_idx] = obj_score

                        if self.pred_contact_threshold == 1.0:
                            history_anno_dict[label][frame_idx] = set(entry['contacting_gt'][rel_idx])
                        else:
                            history_anno_dict[label][frame_idx] = {8} if np.argmax(_contact_distribution[rel_idx]) == 8 else \
                                                                  set(np.where(_contact_distribution[rel_idx] >= self.pred_contact_threshold)[0].tolist()) - {8}

            initialPrior[rel_idx] = self.state_fc(self.initialPrior[label])

            embedding = []
            for c_rel_idx in history_anno_dict[label][frame_idx]:
                embedding.append(self.specificPrior[label][c_rel_idx])
            specificPrior[rel_idx] = self.state_fc(torch.stack(embedding).mean(dim=0)) if len(embedding) != 0 else self.state_fc(self.initialPrior[label])
            

            previous_frame_idx = np.where(history_score_dict[label][:frame_idx].cpu().numpy() != 0)[0].tolist()
            previous_frame_idx = previous_frame_idx[-1] if len(previous_frame_idx) > 0 else frame_idx
            previous_frame_idx = previous_frame_idx if frame_idx - previous_frame_idx <= 2 else frame_idx

            embedding = []
            for c_rel_idx in history_anno_dict[label][previous_frame_idx]:
                embedding.append(self.specificPrior[label][c_rel_idx])
            temporal_priors[rel_idx] = self.state_fc(torch.stack(embedding).mean(dim=0)) if len(embedding) != 0 else self.state_fc(self.initialPrior[label])

        rel_input = torch.zeros([bbox_num * 2, frame_num - 1, features.size(1)]).to(features.device)  # (2 * bbox_num, frame_num - 1, feature_dim)
        rel_prior = torch.zeros([bbox_num * 2, frame_num - 1, features.size(1)]).to(features.device)  # (2 * bbox_num, frame_num - 1, feature_dim)
        position_embed = torch.zeros([bbox_num * 2, frame_num - 1, features.size(1)]).to(features.device)

        idx = -torch.ones([bbox_num * 2, frame_num - 1]).to(features.device)

        for frame_idx in range(frame_num - 1):
            rel_input[:torch.sum((im_idx == frame_idx) + (im_idx == frame_idx + 1)), frame_idx, :] = features[(im_idx == frame_idx) + (im_idx == frame_idx + 1)]

            position_embed[:torch.sum(im_idx == frame_idx), frame_idx, :] = self.position_embedding.weight[0]
            position_embed[torch.sum(im_idx == frame_idx):torch.sum(im_idx == frame_idx)+torch.sum(im_idx == frame_idx+1), frame_idx, :] = self.position_embedding.weight[1]

            if self.use_temporal_prior:
                rel_prior[:torch.sum(im_idx == frame_idx), frame_idx, :] = specificPrior[im_idx == frame_idx] if frame_idx != 0 else initialPrior[im_idx == frame_idx]
                rel_prior[torch.sum(im_idx == frame_idx):torch.sum((im_idx == frame_idx) + (im_idx == frame_idx + 1)), frame_idx, :] = initialPrior[im_idx == frame_idx + 1]

            idx[:torch.sum((im_idx == frame_idx) + (im_idx == frame_idx + 1)), frame_idx] = im_idx[(im_idx == frame_idx) + (im_idx == frame_idx + 1)]

        mask = (torch.sum(rel_input.view(-1, features.size(1)), dim=1) == 0).view(bbox_num * 2, frame_num - 1).permute(1, 0)

        rel_output, attention_weights = self.decoder(rel_input, rel_prior, mask, position_embed)

        temporal_features = torch.zeros_like(features).to(features.device)
        for frame_idx in range(frame_num - 1):
            if frame_idx == 0:
                temporal_features[im_idx == frame_idx] = rel_output[:, frame_idx][idx[:, frame_idx] == frame_idx]

            temporal_features[im_idx == frame_idx + 1] = rel_output[:, frame_idx][idx[:, frame_idx] == frame_idx + 1]

        return temporal_features, temporal_priors



class ensemble_decoder(nn.Module):

    def __init__(self, dec_layer_num=3, embed_dim=1936, nhead=8, dim_feedforward=2048, dropout=0.1, pred_contact_threshold=0.5, window_size=3, 
                 obj_class_num=37, attention_class_num=3, spatial_class_num=6, contact_class_num=17):
        
        super(ensemble_decoder, self).__init__()

        decoder_layer = TransformerDecoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer, dec_layer_num, embed_dim)

        self.position_embedding = nn.Embedding(window_size, embed_dim)
        nn.init.uniform_(self.position_embedding.weight)

        self.pred_contact_threshold = pred_contact_threshold
        self.window_size = window_size

    def forward(self, spatial_features, temporal_features, contact_distribution, im_idx, entry, mode):

        _contact_distribution = contact_distribution.detach().cpu().numpy()

        bbox_num = torch.sum(im_idx == torch.mode(im_idx)[0]) # torch.mode->(tensor, longTensor), 返回众数以及众数下标
        frame_num = int(im_idx[-1] + 1)

        features = torch.zeros([spatial_features.size(0), self.window_size, spatial_features.size(1)]).to(spatial_features.device)
        priors = torch.zeros([spatial_features.size(0), self.window_size, spatial_features.size(1)]).to(spatial_features.device)
        masks = torch.zeros([spatial_features.size(0), self.window_size], dtype=torch.uint8).to(spatial_features.device)
        position_embed = torch.zeros([spatial_features.size(0), self.window_size, spatial_features.size(1)]).to(spatial_features.device)

        history_anno_dict, history_score_dict, history_feature_dict = {}, {}, {}

        for rel_idx in range(spatial_features.size(0)):

            bbox_idx = entry['pair_idx'][rel_idx, 1]
            bbox, label = entry['boxes'][bbox_idx], (int(entry['labels'][bbox_idx].item()) if mode == 'predcls' else int(entry['pred_labels'][bbox_idx].item()))
            obj_score = entry['pred_scores'][bbox_idx] if (mode != 'predcls') and (not self.training) else 1
            frame_idx = int(entry['im_idx'][rel_idx].item())

            if label not in history_anno_dict.keys():
                history_anno_dict[label] = [None for _ in range(frame_num)]
                history_score_dict[label] = torch.zeros([frame_num]).to(spatial_features.device)
                history_feature_dict[label] = torch.zeros([frame_num, spatial_features.size(1)]).to(spatial_features.device)

            #history_feature_dict[label][frame_idx] = torch.cat((spatial_features[rel_idx], temporal_features[rel_idx]))

            if self.training:
                history_score_dict[label][frame_idx] = 1.0
                history_feature_dict[label][frame_idx] = (spatial_features[rel_idx] + temporal_features[rel_idx]) / 2 #torch.cat((spatial_features[rel_idx], temporal_features[rel_idx]))
                
                if self.pred_contact_threshold == 1.0:
                    history_anno_dict[label][frame_idx] = set(entry['contacting_gt'][rel_idx])
                else:
                    history_anno_dict[label][frame_idx] = {8} if np.argmax(_contact_distribution[rel_idx]) == 8 else \
                                                          set(np.where(_contact_distribution[rel_idx] >= self.pred_contact_threshold)[0].tolist()) - {8}
            else:
                if mode == 'predcls':
                    history_score_dict[label][frame_idx] = 1.0
                    history_feature_dict[label][frame_idx] = (spatial_features[rel_idx] + temporal_features[rel_idx]) / 2 #torch.cat((spatial_features[rel_idx], temporal_features[rel_idx]))

                    if self.pred_contact_threshold == 1.0:
                        history_anno_dict[label][frame_idx] = set(entry['contacting_gt'][rel_idx])
                    else:
                        history_anno_dict[label][frame_idx] = {8} if np.argmax(_contact_distribution[rel_idx]) == 8 else \
                                                              set(np.where(_contact_distribution[rel_idx] >= self.pred_contact_threshold)[0].tolist()) - {8}
                else:
                    if obj_score > history_score_dict[label][frame_idx]:
                        history_score_dict[label][frame_idx] = obj_score
                        history_feature_dict[label][frame_idx] = (spatial_features[rel_idx] + temporal_features[rel_idx]) / 2 #torch.cat((spatial_features[rel_idx], temporal_features[rel_idx]))

                        if self.pred_contact_threshold == 1.0:
                            history_anno_dict[label][frame_idx] = set(entry['contacting_gt'][rel_idx])
                        else:
                            history_anno_dict[label][frame_idx] = {8} if np.argmax(_contact_distribution[rel_idx]) == 8 else \
                                                                  set(np.where(_contact_distribution[rel_idx] >= self.pred_contact_threshold)[0].tolist()) - {8}

            # Pick key frame
            candidate_idxes = [frame_idx] 
            for idx in np.where(history_score_dict[label][:frame_idx].cpu().numpy() != 0)[0].tolist()[::-1]:
                if candidate_idxes[-1] - idx <= 2:
                    candidate_idxes.append(idx)
                if len(candidate_idxes) == self.window_size:
                    break
            candidate_idxes = candidate_idxes[::-1]

            masks[rel_idx, len(candidate_idxes):] = 1
            for _idx, candidate_idx in enumerate(candidate_idxes):    
                position_embed[rel_idx, _idx, :] = self.position_embedding.weight[_idx]
                features[rel_idx, _idx] = history_feature_dict[label][candidate_idx]

        output, attention_weights = self.decoder(features.permute(1, 0, 2), priors.permute(1, 0, 2), masks, position_embed.permute(1, 0, 2))

        ensemble_features = torch.zeros((spatial_features.size(0), spatial_features.size(1))).to(spatial_features.device)
        for rel_idx in range(spatial_features.size(0)):
            ensemble_features[rel_idx] = output[int(self.window_size-sum(masks[rel_idx])-1), rel_idx]

        return ensemble_features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
