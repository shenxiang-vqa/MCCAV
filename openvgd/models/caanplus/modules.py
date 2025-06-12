import torch
import numpy as np
import math, random, json
import torch.nn as nn
import torch.nn.functional as F

''' 
==================
    Operations
==================
'''

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()
        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6, dim=-1):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.dim = dim
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        std = x.std(self.dim, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    
class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HSIZE,
            mid_size=__C.HSIZE * 4,
            out_size=__C.HSIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)
    
class FFN1(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



    
class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HSIZE,
            mid_size=__C.ATTFLAT_MLP_SIZE,
            out_size=__C.ATTFLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.linear_merge = nn.Linear(__C.HSIZE * __C.ATTFLAT_GLIMPSES, __C.ATTFLAT_OUT_SIZE)

    def forward(self, x, x_mask=None):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.ATTFLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class MHAtt(nn.Module):
    def __init__(self, __C, base=64, hsize_k=None, bias=False):
        super(MHAtt, self).__init__()
        self.__C = __C
        self.HBASE = base

        if hsize_k:
            self.HSIZE_INSIDE = int(__C.HSIZE * hsize_k)
        else:
            self.HSIZE_INSIDE = __C.HSIZE

        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)

        self.linear_v = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_k = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_q = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_merge = nn.Linear(self.HSIZE_INSIDE, __C.HSIZE, bias=bias)
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.HSIZE_INSIDE)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

    
# -----------------------------------
# --Absoluate position modulete MHA--
# -----------------------------------
class ABS_MHAtt(nn.Module):
    def __init__(self, __C, base=64, hsize_k=None, bias=False):
        super(ABS_MHAtt, self).__init__()
        self.__C = __C
        self.HBASE = base

        if hsize_k:
            self.HSIZE_INSIDE = int(__C.HSIZE * hsize_k)
        else:
            self.HSIZE_INSIDE = __C.HSIZE

        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)

        self.linear_v = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_k = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_q = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_merge = nn.Linear(self.HSIZE_INSIDE, __C.HSIZE, bias=bias)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        #self.LN = LayerNorm(self.HBASE)

    def forward(self, v, k, q, abs_mask, mask, img_abs):
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        img_abs = img_abs.view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        
        q = self.modulate(q, img_abs, abs_mask)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.HSIZE_INSIDE)
        atted = self.linear_merge(atted)

        return atted

    
    def modulate(self, query, pos, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, pos.transpose(-2, -1)
        ) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return query + torch.matmul(att_map, pos)
    
    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# -----------------------------------
# ---ABS_REL position modulete MHA---
# -----------------------------------

class ABS_REL_MHAtt(nn.Module):
    def __init__(self, __C, base=64, hsize_k=None, bias=False):
        super(ABS_REL_MHAtt, self).__init__()
        self.__C = __C
        self.HBASE = base

        if hsize_k:
            self.HSIZE_INSIDE = int(__C.HSIZE * hsize_k)
        else:
            self.HSIZE_INSIDE = __C.HSIZE

        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)

        self.linear_v = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_k = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_q = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_tk = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_tv = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        #self.LN = LayerNorm(self.HBASE)
        
        if self.__C.REL == 'True':
            self.WGs = nn.ModuleList([nn.Linear(96, 1, bias=True) for _ in range(self.HHEAD)])
            
        self.linear_merge = nn.Linear(self.HSIZE_INSIDE, __C.HSIZE, bias=bias)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.ffn = FFN1(64,64,64,1)

    def forward(self, v, k, q, t, mask, t_mask, img_abs, img_rel):
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        img_abs = img_abs.view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        t_k = self.linear_tk(t).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        t_v = self.linear_tv(t).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        
        t1 = self.att(t_v, t_k, q, t_mask)
        q = self.modulate(q, img_abs, mask)
        k = self.modulate(k, img_abs, mask)
        
        q = self.Verify_score(t1, q)
        
        if self.__C.REL == 'True':
            flatten_relative_geometry_embeddings = img_rel.view(-1, 96)
            box_size_per_head = list(img_rel.shape[:3])
            box_size_per_head.insert(1, 1)
            relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in self.WGs]
            relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
            w_g = F.relu(relative_geometry_weights)
            atted = self.att_with_rel(v, k, q, w_g, mask)
        else: 
            atted = self.att(v, k, q, mask)
            
        
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.HSIZE_INSIDE)
        atted = self.linear_merge(atted)

        return atted
    
    def modulate(self, query, pos, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, pos.transpose(-2, -1)
        ) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return query + torch.matmul(att_map, pos)
    
    def att_with_rel(self, value, key, query, w_g, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        
        scores = scores + torch.log(torch.clamp(w_g, min=1e-6))
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value) 
    
    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
        
        
    # V_L 验证模块
    def Verify_score(self, vl_text, vl_query):
        text_embed = self.ffn(vl_text)
        #print(text_embed.size())
        img_embed = self.ffn(vl_query)
        scale = 1.0
        sigma = 0.5
        p = 2.0
        verify_s = (F.normalize(img_embed, p=2, dim=-1) *
                    F.normalize(text_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        verify_s = scale * \
                   torch.exp(- (1 - verify_s).pow(int(p)) \
                             / (2 * sigma ** 2))
        vl_q = verify_s * vl_query

        return vl_q


    def Verify_jaccard(self, vl_text, vl_query):
       
        text_embed = self.ffn(vl_text) #[64,8,100,64]
        #print('text',text_embed.size())
        img_embed = self.ffn(vl_query) #[64,8,100,64]
        #print('img',img_embed.size())
        # 计算Jaccard相似性
        intersection = torch.min(text_embed, img_embed)
        union = torch.max(text_embed, img_embed)
        
        # 将相似性映射到[0, 1]范围
        epsilon = 1e-8  # 用于防止除以零
        jaccard_similarity = (intersection.sum(dim=-1, keepdim=True) + epsilon) / (union.sum(dim=-1, keepdim=True) + epsilon)

        vl_q = jaccard_similarity * vl_query  # [64,8,100,64]
                            
        #vl_q = verify_s * vl_query #[64,8,100,64]
        #print(vl_q.size())

        return vl_q
                         

# ------------------------------
# ---- Global_Context MHA ----
# ------------------------------

class GMHAtt(nn.Module):
    def __init__(self, __C, base=64, hsize_k=None, bias=False):
        super(GMHAtt, self).__init__()
        self.__C = __C
        self.HBASE = base

        if hsize_k:
            self.HSIZE_INSIDE = int(__C.HSIZE * hsize_k)
        else:
            self.HSIZE_INSIDE = __C.HSIZE

        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)

        self.linear_v = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_k = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_q = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_merge = nn.Linear(self.HSIZE_INSIDE, __C.HSIZE, bias=bias)
        self.avgpool_k = nn.AdaptiveAvgPool2d((1,None))
        self.avgpool_q = nn.AdaptiveAvgPool2d((1,None))
        self.lin_uk = nn.Linear(__C.HSIZE, __C.HSIZE, bias=bias)
        self.lin_uq = nn.Linear(__C.HSIZE, __C.HSIZE, bias=bias)
        self.tran_q = nn.Linear(__C.HSIZE, 1, bias=bias)
        self.tran_k = nn.Linear(__C.HSIZE, 1, bias=bias)
        self.tran_cq = nn.Linear(__C.HSIZE, 1, bias=bias)
        self.tran_ck = nn.Linear(__C.HSIZE, 1, bias=bias)
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask=None, c=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        
        c_k = self.lin_uk(self.avgpool_k(k)) # (B, 1, 512)
        c_q = self.lin_uq(self.avgpool_q(q))

        k = self.linear_k(k)  # (B, N, 512)
        q = self.linear_q(q)

        merge_q = self.tran_q(q) + self.tran_cq(c_q) # (B. N, 1)
        lamta_q = torch.sigmoid(merge_q)

        merge_k = self.tran_k(k) + self.tran_ck(c_k)
        lamta_k = torch.sigmoid(merge_k)

        q = (1-lamta_q) * q + lamta_q * c_q # (B, N, 512)
        k = (1-lamta_k) * k + lamta_k * c_k

        k = k.view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        q = q.view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.HSIZE_INSIDE)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
    
# ------------------------------   
# ------ Deep-context MHA ------
# ------------------------------

class DMHAtt(nn.Module):
    def __init__(self, __C, l_layer, base=64, hsize_k=None, bias=False):
        super(DMHAtt, self).__init__()
        self.__C = __C
        self.HBASE = base

        if hsize_k:
            self.HSIZE_INSIDE = int(__C.HSIZE * hsize_k)
        else:
            self.HSIZE_INSIDE = __C.HSIZE

        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)
        
        self.lin_layer = nn.Linear(__C.HSIZE * l_layer, __C.HSIZE, bias=bias)

        self.linear_v = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_k = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_q = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_merge = nn.Linear(self.HSIZE_INSIDE, __C.HSIZE, bias=bias)
        self.lin_uk = nn.Linear(__C.HSIZE, __C.HSIZE, bias=bias)
        self.lin_uq = nn.Linear(__C.HSIZE, __C.HSIZE, bias=bias)
        self.tran_q = nn.Linear(__C.HSIZE, 1, bias=bias)
        self.tran_k = nn.Linear(__C.HSIZE, 1, bias=bias)
        self.tran_cq = nn.Linear(__C.HSIZE, 1, bias=bias)
        self.tran_ck = nn.Linear(__C.HSIZE, 1, bias=bias)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, c):
        n_batches = q.size(0)

        if c.__len__() == 1:
            c = c[0]
        else:
            c = torch.cat(c, dim=-1)
        c = self.lin_layer(c) # (B, N, 512)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)

        c_k = self.lin_uk(c) # (B, N, 512)
        c_q = self.lin_uq(c)

        k = self.linear_k(k)  # (B, N, 512)
        q = self.linear_q(q)

        merge_q = self.tran_q(q) + self.tran_cq(c_q) # (B. N, 1)
        lamta_q = torch.sigmoid(merge_q)

        merge_k = self.tran_k(k) + self.tran_ck(c_k)
        lamta_k = torch.sigmoid(merge_k)

        q = (1-lamta_q) * q + lamta_q * c_q # (B, N, 512)
        k = (1-lamta_k) * k + lamta_k * c_k

        k = k.view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        q = q.view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.HSIZE_INSIDE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# --------------------------------
# ---- DeepGlobal-context MHA ----
# --------------------------------

class DGMHAtt(nn.Module):
    def __init__(self, __C, l_layer, base=64, hsize_k=None, bias=False):
        super(DGMHAtt, self).__init__()
        self.__C = __C
        self.HBASE = base

        if hsize_k:
            self.HSIZE_INSIDE = int(__C.HSIZE * hsize_k)
        else:
            self.HSIZE_INSIDE = __C.HSIZE

        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)
        
        self.lin_layer = nn.Linear(__C.HSIZE * l_layer, __C.HSIZE, bias=bias)

        self.linear_v = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_k = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_q = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_merge = nn.Linear(self.HSIZE_INSIDE, __C.HSIZE, bias=bias)
        self.lin_uk = nn.Linear(__C.HSIZE, __C.HSIZE, bias=bias)
        self.lin_uq = nn.Linear(__C.HSIZE, __C.HSIZE, bias=bias)
        self.tran_q = nn.Linear(__C.HSIZE, 1, bias=bias)
        self.tran_k = nn.Linear(__C.HSIZE, 1, bias=bias)
        self.tran_cq = nn.Linear(__C.HSIZE, 1, bias=bias)
        self.tran_ck = nn.Linear(__C.HSIZE, 1, bias=bias)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, c):
        n_batches = q.size(0)

        if c.__len__() == 1:
            c = c[0]
        else:
            c = torch.cat(c, dim=-1)
        c = self.lin_layer(c) # (B, 1, 512)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)

        c_k = self.lin_uk(c) # (B, 1, 512)
        c_q = self.lin_uq(c)

        k = self.linear_k(k)  # (B, 1, 512)
        q = self.linear_q(q)

        merge_q = self.tran_q(q) + self.tran_cq(c_q) # (B. N, 1)
        lamta_q = torch.sigmoid(merge_q)

        merge_k = self.tran_k(k) + self.tran_ck(c_k)
        lamta_k = torch.sigmoid(merge_k)

        q = (1-lamta_q) * q + lamta_q * c_q # (B, N, 512)
        k = (1-lamta_k) * k + lamta_k * c_k
        
        k = k.view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        q = q.view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.HSIZE_INSIDE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C, l_layer):
        super(SA, self).__init__()
        self.__C = __C
        
        if self.__C.USE_CONTEXT == 'None':
            self.mhatt = MHAtt(__C, base=64, hsize_k=None)
        
        elif self.__C.USE_CONTEXT == 'global':
            self.mhatt = GMHAtt(__C, base=64, hsize_k=None)
            
        elif self.__C.USE_CONTEXT == 'deep':
            self.mhatt = DMHAtt(__C, l_layer = l_layer, base=64, hsize_k=None)
            
        elif self.__C.USE_CONTEXT == 'deep-global':
            self.mhatt = DGMHAtt(__C, l_layer = l_layer, base=64, hsize_k=None)
            
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HSIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HSIZE)

    def forward(self, y, y_mask, c):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask, c)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()
        
        self.mhatt1 = ABS_REL_MHAtt(__C, base=64, hsize_k=None)
            
        self.mhatt2 = ABS_MHAtt(__C, base=64, hsize_k=None)

        #self.mhatt1 = MHAtt(__C, base=64, hsize_k=None)
        #self.mhatt2 = MHAtt(__C, base=64, hsize_k=None)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HSIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HSIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HSIZE)
        
    def forward(self, x, y, x_mask, y_mask, img_abs, img_rel):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, t=y, mask=x_mask, t_mask=y_mask, img_abs=img_abs, img_rel=img_rel)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, abs_mask=x_mask, mask=y_mask, img_abs=img_abs)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))
        return x


# ------------------------------------------------
# --CAANPLUS Layers Cascaded by Encoder-Decoder --
# ------------------------------------------------

class CAANPLUS_ED(nn.Module):
    def __init__(self, __C):
        super(CAANPLUS_ED, self).__init__()       
        self.__C = __C

        self.enc_list = nn.ModuleList([SA(__C, l_layer = i+1) for i in range(6)])
        self.dec_list = nn.ModuleList([SGA(__C) for i in range(6)])
        
        if __C.USE_CONTEXT == 'deep-global':
            self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        

    def forward(self, y, x, y_mask, x_mask, img_abs, img_rel):
        # Get encoder last hidden vector
        
        if self.__C.USE_CONTEXT == 'None':
            for enc in self.enc_list:
                y = enc(y, y_mask, c=None)
        
        elif self.__C.USE_CONTEXT == 'deep':
            c = [y]
            for enc in self.enc_list:
                y = enc(y, y_mask, c)
                c.append(y)
                
        elif self.__C.USE_CONTEXT == 'global':
            for enc in self.enc_list:
                y = enc(y, y_mask, c=None)
                
        elif self.__C.USE_CONTEXT == 'deep-global':
            # use other methods can replace avgpool. 
            c = [self.avgpool(y)]
            for enc in self.enc_list:
                y = enc(y, y_mask, c)
                c.append(self.avgpool(y))

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask, img_abs, img_rel)

        return y, x
