import torch
from torch import nn
from torchcrf import CRF
from .img_extraction_module import Focus, Conv, BottleneckCSP, SPP, Concat
from .spatial_channel_attention import SpatialAttention, ChannelAttention
from transformers import BertModel
import torch.nn.functional as F
import numpy as np

# 112 224 336 448
# G1&G2 G2&G3 G1&G3
class ImageModel(nn.Module):
    def __init__(self, hidden_size, negative_slope1, negative_slope2, k=10):
        super(ImageModel, self).__init__()
        self.hidden_size = hidden_size
        self.focus = Focus(c1=3, c2=64, k=3, negative_slope=negative_slope2)
        self.cbl_1 = Conv(c1=64, c2=128, k=3, s=2, negative_slope=negative_slope1)
        self.csp_1 = BottleneckCSP(c1=128, c2=128, n=3, negative_slope=negative_slope2)

        self.cbl_2 = Conv(c1=128, c2=256, k=3, s=2, negative_slope=negative_slope1)
        self.csp_2 = BottleneckCSP(c1=256, c2=256, n=6, negative_slope=negative_slope2)  # layer4

        self.cbl_3 = Conv(c1=256, c2=512, k=3, s=2, negative_slope=negative_slope1)
        self.csp_3 = BottleneckCSP(c1=512, c2=512, n=9, negative_slope=negative_slope2)  # layer6

        self.cbl_4 = Conv(c1=512, c2=1024, k=3, s=2, negative_slope=negative_slope1)
        self.csp_4 = BottleneckCSP(c1=1024, c2=1024, n=3, negative_slope=negative_slope2)
        self.ssp = SPP(c1=1024, c2=1024, k=[5, 9, 13], negative_slope=negative_slope2)

        self.spatial_atten = SpatialAttention(in_planes=1024, text_dim=hidden_size, k=k)
        self.channel_atten = ChannelAttention(in_planes=1024, text_dim=hidden_size, k=k)

        self.cbl_5 = Conv(c1=1024, c2=512, k=1, s=1, negative_slope=negative_slope1)
        self.upsample1 = nn.Upsample(size=None, scale_factor=2, mode='nearest')
        self.concat1 = Concat(1)
        self.csp_5 = BottleneckCSP(c1=512 + 512, c2=512, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_6 = Conv(c1=512, c2=256, k=1, s=1, negative_slope=negative_slope1)
        self.upsample2 = nn.Upsample(size=None, scale_factor=2, mode='nearest')
        self.concat2 = Concat(1)
        self.csp_6 = BottleneckCSP(c1=256 + 256, c2=256, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_7 = Conv(c1=256, c2=256, k=3, s=2, negative_slope=negative_slope1)
        self.concat3 = Concat(1)
        self.csp_7 = BottleneckCSP(c1=256 + 256, c2=512, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_8 = Conv(c1=512, c2=512, k=3, s=2, negative_slope=negative_slope1)
        self.concat4 = Concat(1)
        self.csp_8 = BottleneckCSP(c1=512 + 512, c2=1024, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_final = Conv(c1=1024, c2=self.hidden_size, k=1, s=1, negative_slope=negative_slope1)
        self.csp_final = BottleneckCSP(c1=self.hidden_size, c2=self.hidden_size, n=3, shortcut=True, negative_slope=negative_slope2)

    def forward(self, x, text_feat):
        layer4 = self.csp_2(self.cbl_2(self.csp_1(self.cbl_1(self.focus(x)))))
        layer6 = self.csp_3(self.cbl_3(layer4))
        layer9 = self.ssp(self.csp_4(self.cbl_4(layer6)))

        layer10 = self.cbl_5(layer9) # G3
        layer13 = self.csp_5(self.concat1([self.upsample1(layer10), layer6]))

        layer14 = self.cbl_6(layer13)   # G2
        layer17 = self.csp_6(self.concat2([self.upsample2(layer14), layer4]))     # G1

        layer20 = self.csp_7(self.concat3([self.cbl_7(layer17), layer14]))

        layer23 = self.csp_8(self.concat4([self.cbl_8(layer20), layer10]))

        layer23 = layer23.mul(self.channel_atten(layer23, text_feat))
        layer23 = layer23.mul(self.spatial_atten(layer23, text_feat))

        final_layer = self.csp_final(self.cbl_final(layer23))

        return final_layer


class FANetModel(nn.Module):
    def __init__(self, label_list, args):
        super(FANetModel, self).__init__()
        self.args = args
        self.num_labels = len(label_list)  # 13

        self.text_encoder = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.text_encoder.config

        self.vision_encoder = ImageModel(
            self.bert_config.hidden_size, 
            args.negative_slope1, 
            args.negative_slope2, 
            args.dyn_k
        )
        self.vision_linear = nn.Linear(
            in_features=7 * 7,
            out_features=1
        )

        self.text_proj = nn.Linear(
            in_features=self.bert_config.hidden_size,
            out_features=self.args.embed_dim
        )
        self.vision_proj = nn.Linear(
            in_features=self.bert_config.hidden_size,
            out_features=self.args.embed_dim
        )

        self.temp = nn.Parameter(torch.ones([]) * self.args.temp)     # temp
        self.queue_size = self.args.queue_size
        self.momentum = self.args.momentum
        self.loss_itc = 0.0

        # create momentum model
        self.text_encoder_m = BertModel.from_pretrained(args.bert_name)
        self.text_proj_m = nn.Linear(
            in_features=self.bert_config.hidden_size,
            out_features=self.args.embed_dim
        )
        self.vision_encoder_m = ImageModel(
            self.bert_config.hidden_size, 
            args.negative_slope1, 
            args.negative_slope2, 
            args.dyn_k
        )
        self.vision_linear_m = nn.Linear(
            in_features=7 * 7,
            out_features=1
        )
        self.vision_proj_m = nn.Linear(
            in_features=self.bert_config.hidden_size,
            out_features=self.args.embed_dim
        )

        self.model_pairs = [
            [self.vision_encoder, self.vision_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m]
        ]

        self.copy_params()

        # create the queue
        self.register_buffer('vision_queue', torch.randn(self.args.embed_dim, self.queue_size))
        self.register_buffer('text_queue', torch.randn(self.args.embed_dim, self.queue_size))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.vision_queue = F.normalize(input=self.vision_queue, dim=0)
        self.text_queue = F.normalize(input=self.text_queue, dim=0)

        self.Cross_atten = MultiHeadAttention(
            n_head=self.bert_config.num_attention_heads, 
            d_model=self.bert_config.hidden_size, 
            d_k=self.bert_config.hidden_size, 
            d_v=self.bert_config.hidden_size
        )

        self.classifier = nn.Linear(self.bert_config.hidden_size, self.num_labels)
        self.final_dropout = nn.Dropout(args.dropout_prob)

        self.crf = CRF(self.num_labels, batch_first=True)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, alpha=0, mode='train'):
        # text encoder, output are last_hidden_state and pooler_output
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        text_hidden_state, text_pooler_output = text_output.last_hidden_state, text_output.pooler_output
        text_global_info = self.compute_global_info(text_pooler_output.unsqueeze(1), text_hidden_state)
        text_feat = F.normalize(self.text_proj(text_global_info), dim=-1)

        # vision encoder, output are last_hidden_state and pooler_output
        vision_output = self.vision_encoder(images, text_pooler_output) # [bsz, 768, 7, 7]
        vision_hidden_state = vision_output.flatten(2).permute(0, 2, 1)
        vision_pooler_output = self.vision_linear(vision_output.flatten(2)).view(vision_output.size(0), -1).unsqueeze(1)
        vision_global_info = self.compute_global_info(vision_pooler_output, vision_hidden_state)
        vision_feat = F.normalize(self.vision_proj(vision_global_info), dim=-1)

        # if mode is train, use momentum model for Global Contrastive Learning
        if mode == 'train':
            # get momentum features
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)  # 设置temp值区间范围min, max
                self._momentum_update()
                text_output_m = self.text_encoder_m(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_dict=True
                )
                text_hidden_state_m, text_pooler_output_m = text_output_m.last_hidden_state, \
                    text_output_m.pooler_output

                momentum_labels = []
                for line_label in labels:
                    momentum_labels.append([1 if label >= 2 and label <= 10 else 0 for label in line_label])
                momentum_labels = torch.tensor(momentum_labels).to(self.args.device)

                entity_hidden_state_m = torch.mul(momentum_labels.unsqueeze(2), text_hidden_state_m)

                text_global_info_m = self.compute_global_info(text_pooler_output_m.unsqueeze(1), entity_hidden_state_m)
                text_feat_m = F.normalize(self.text_proj_m(text_global_info_m), dim=-1)
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

                vision_output_m = self.vision_encoder_m(images, text_pooler_output_m)
                vision_hidden_state_m = vision_output_m.flatten(2).permute(0, 2, 1)
                vision_pooler_output_m = self.vision_linear_m(vision_output_m.flatten(2)).view(vision_output_m.size(0), -1).unsqueeze(1)
                vision_global_info_m = self.compute_global_info(vision_pooler_output_m, vision_hidden_state_m)
                vision_feat_m = F.normalize(self.vision_proj_m(vision_global_info_m), dim=-1)
                vision_feat_all = torch.cat([vision_feat_m.t(), self.vision_queue.clone().detach()], dim=1)

                sim_i2t_m = vision_feat_m @ text_feat_all / self.temp

                sim_targets = torch.zeros(sim_i2t_m.size()).to(self.args.device)
                sim_targets.fill_diagonal_(1)

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets

            sim_i2t = vision_feat @ text_feat_all / self.temp

            # compute KL loss
            self.loss_itc = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()

            self._dequeue_and_enqueue(vision_feat_m, text_feat_m)

        # Cross-Modal Multi-head Attention
        cross_output, _ = self.Cross_atten(text_hidden_state, vision_hidden_state, vision_hidden_state)

        # full-linear classifier
        emissions = self.final_dropout(self.classifier(cross_output))

        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        # if labels is not None:
        loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
        if mode == 'train':
            return logits, loss, self.loss_itc
        return logits, loss

    def compute_global_info(self, input_cls, input_hidden_state):
        per_weights = torch.sum(torch.mul(input_cls, input_hidden_state), dim=-1, keepdim=True)
        per_weights = torch.nn.Softmax(dim=1)(per_weights)
        global_info = torch.sum(torch.mul(per_weights, input_hidden_state), dim=1, keepdim=False)
        return global_info

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats):
        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.vision_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False, attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            # self.dropout2 = nn.Dropout(dropout2)
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None, dec_self=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        if hasattr(self, 'dropout2'):
            q = self.dropout2(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        if hasattr(self, 'fc'):
            output = self.fc(output)

        if hasattr(self, 'dropout'):
            output = self.dropout(output)

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e6)

        if stop_sig:
            print('**')
            stop()

        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
