import math
import torch
from torch import nn
from torch.nn import functional as F
from extractor import PhyModel_1,CNN_1D_P1,CNN_1D_V1
class PIMFuseModel(nn.Module):
    def __init__(self,num_classes,hidden_size,logit_average=False):
        super().__init__()
        self.num_classes = num_classes
        self.logit_average = logit_average
        self.vibration_model = CNN_1D_V1(input_channels=3,num_classes=num_classes)
        self.pressure_model = CNN_1D_P1(input_channels=1, num_classes=num_classes)
        self.physical_model = PhyModel_1(input_channels=1, num_classes=num_classes)
        self.shared_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.fuse_model_shared = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.attn_proj = nn.Linear(hidden_size, (2 + num_classes) * hidden_size)
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)
    def forward(self, pairs, S_V, S_P,S_P1):
        S_V=S_V.permute(0, 2, 1)
        S_P=S_P.permute(0, 2, 1)
        S_P1=S_P1.permute(0, 2, 1)
        feat_vibration_shared, feat_vibration_distinct, pred_vibration = self.vibration_model(S_V)
        feat_pressure_shared, feat_pressure_distinct, pred_pressure = self.pressure_model(S_P)
        feat_vibration_shared = self.shared_project(feat_vibration_shared)
        feat_pressure_shared = self.shared_project(feat_pressure_shared)
        y_30,y_pred_phy= self.physical_model(S_P1)

        pairs = pairs.unsqueeze(1)

        h1 = feat_vibration_shared
        h2 = feat_pressure_shared
        term1 = torch.stack([h1 + h2, h1 + h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)
        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_pressure_shared
        pred_shared = self.fuse_model_shared(feat_avg_shared)

        attn_input = torch.stack([feat_pressure_distinct,feat_avg_shared,y_30,feat_vibration_distinct], dim=1)
        qkvs = self.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2 + self.num_classes, dim=-1)
        q_mean = pairs * q.mean(dim=1) + (1 - pairs) * q[:, :-1].mean(dim=1)
        ks = torch.stack(k, dim=1)
        attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)
        attn_logits = attn_logits / math.sqrt(q.shape[-1])
        attn_mask = torch.ones_like(attn_logits)
        attn_mask[pairs.squeeze() == 0, :,-1] = 0
        attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)

        feat_final = torch.matmul(attn_weights, v)
        pred_final = self.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2)
        outputs = {
            'feat_vibration_shared': feat_vibration_shared,
            'feat_pressure_shared': feat_pressure_shared,
            'feat_vibration_distinct': feat_vibration_distinct,
            'feat_pressure_distinct': feat_pressure_distinct,
            'feat_final': feat_final,
            'pred_final': pred_final,
            'pred_shared': pred_shared,
            'pred_vibration': pred_vibration,
            'pred_pressure': pred_pressure,
            'attn_weights': attn_weights,
            'y_pred_phy': y_pred_phy,
            'y_30': y_30,
            'S_P1': S_P1
        }
        return outputs






