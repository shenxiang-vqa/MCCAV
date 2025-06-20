import torch
import torch.nn as nn
import torch.nn.functional as F
from openvgd.models.caanplus.modules import AttFlat, LayerNorm, CAANPLUS_ED
from openvgd.models.caanplus.position import RegionAbsolutePosition, RegionRelationalEmbedding

class Net_Full(nn.Module):
    def __init__(self, __C, init_dict):
        super(Net_Full, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(num_embeddings=init_dict['token_size'], embedding_dim=__C.WORD_EMBED_SIZE)
        self.embedding.weight.data.copy_(torch.from_numpy(init_dict['pretrained_emb']))
        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HSIZE,
            num_layers=1,
            batch_first=True
        )

        imgfeat_linear_size = __C.FRCNFEAT_SIZE
        if __C.BBOX_FEATURE:
            self.bboxfeat_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.imgfeat_linear = nn.Linear(imgfeat_linear_size, __C.HSIZE)
        self.bbox_linear = nn.Linear(6, __C.HSIZE)

        self.backnone = CAANPLUS_ED(__C)
        self.attflat_x = AttFlat(__C)
        self.attfc_y = nn.Linear(__C.HSIZE, __C.ATTFLAT_OUT_SIZE)
        self.proj_norm = LayerNorm(__C.ATTFLAT_OUT_SIZE)
        self.proj_scores = nn.Linear(__C.ATTFLAT_OUT_SIZE, 1)
        self.proj_reg = nn.Linear(__C.ATTFLAT_OUT_SIZE, 4)
        #self.linear_y_rel = nn.Linear(4, __C.REL_SIZE)
        #self.linear_y_rel = nn.Linear(6, __C.REL_SIZE)


    def forward(self, input):
        frcn_feat, bbox_feat, bbox_abs, y_rel_embed, ques_ix, x_rel_embed, bbox, img_shape= input

        # with torch.no_grad():
        # Make mask for attention learning
        x_mask = self.make_mask(ques_ix.unsqueeze(2))
        y_mask = self.make_mask(frcn_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        x_in, _ = self.lstm(lang_feat)
        
        region_abs = RegionAbsolutePosition(bbox, img_shape)

        # Pre-process Image Feature
        if self.__C.BBOX_FEATURE:
            bbox_feat = self.bboxfeat_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        y_in = self.imgfeat_linear(frcn_feat)
        
        region_abs = self.bbox_linear(region_abs)    # (bs, num, 512)
        region_rel = RegionRelationalEmbedding(bbox) # (bs, num, num, 6 or 96)

        #y_rel_embed = F.relu(self.linear_y_rel(y_rel_embed))
        x_out, y_out = self.backnone(x_in, y_in, x_mask, y_mask, region_abs, region_rel)
        x_out = self.attflat_x(x_out, x_mask).unsqueeze(1)
        y_out = self.attfc_y(y_out)
        xy_out = x_out + y_out
        xy_out = self.proj_norm(xy_out)
        pred_scores = self.proj_scores(xy_out).squeeze(-1)
        if self.__C.SCORES_LOSS == 'kld':
            pred_scores = F.log_softmax(pred_scores, dim=-1)
        pred_reg = self.proj_reg(xy_out)

        return pred_scores, pred_reg

    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)