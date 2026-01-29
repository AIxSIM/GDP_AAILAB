import torch 
import torch.nn as nn
import torch.nn.functional as F 
from einops import rearrange
from einops.layers.torch import Rearrange
import pickle

from models_seq.blocks import (
    SinusoidalPosEmb, 
    Conv1dBlock, 
    Residual, 
    LinearAttention, 
)


class XTResBlock(nn.Module):

    def __init__(self, x_in_dim, t_in_dim, out_dim, device, kernel_size=5):
        super().__init__()
        self.device = device
        self.block1 = Conv1dBlock(x_in_dim, out_dim, kernel_size).to(device)
        self.block2 = Conv1dBlock(out_dim, out_dim, kernel_size).to(device)

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(t_in_dim, out_dim, device=device),
            Rearrange("b c -> b 1 c") 
        )
        self.residual_conv = nn.Sequential(
            Rearrange("b h c -> b c h"), 
            nn.Conv1d(x_in_dim, out_dim, 1, device=device) if x_in_dim != out_dim else nn.Identity(), 
            Rearrange("b c h -> b h c")
        )

    def forward(self, x, t):
        '''
            x : b h c
            t : b h d
            returns:
            out : b h e
        '''
        out = self.block1(x) + self.time_mlp(t)
        out = self.block2(out)
        return out + self.residual_conv(x)
    
    
class UnetBlock(nn.Module):
    def __init__(self, x_dim, time_dim, out_dim, device, down_up, last):
        super().__init__()
        self.device = device
        self.xtblock1 = XTResBlock(x_dim, time_dim, out_dim, device, kernel_size=5)
        self.xtblock2 = XTResBlock(out_dim, time_dim, out_dim, device, kernel_size=5)
        self.attn = Residual(LinearAttention(out_dim, device))
        self.down = down_up == "down"
        self.sample = nn.Identity()
        
    def forward(self, xs, lengths, ts):
        x = self.xtblock1(xs, ts)
        x = self.xtblock2(x, ts)
        h = self.attn(x, lengths)
        x = self.sample(h)
        return x, h


class Discriminator(nn.Module):
    def __init__(
        self,
        n_vertex,
        x_emb_dim,
        dims,
        hidden_dim,
        device,
        pretrain_path=None,
        use_mid_attn=False,     # 필요하면 True
        cls_hidden=None,        # None이면 mid_dim 그대로, 값 주면 더 작게
    ):
        super().__init__()
        self.device = device
        time_dim = hidden_dim

        # temporal embedding (그대로 두되 Mish -> SiLU로 조금 더 가볍게)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim, device),
            nn.Linear(time_dim, 4 * time_dim, device=device),
            nn.SiLU(),
            nn.Linear(4 * time_dim, time_dim, device=device),
        )

        # embedding
        if pretrain_path is not None:
            import pickle
            node2vec = pickle.load(open(pretrain_path, "rb"))
            assert n_vertex == len(node2vec)
            x_emb_dim = node2vec[0].shape[0]

            nodeemb = torch.zeros(n_vertex + 2, x_emb_dim)
            for k in node2vec:
                nodeemb[k] = torch.from_numpy(node2vec[k])
            self.x_embedding = nn.Embedding.from_pretrained(nodeemb, freeze=False).to(device)
        else:
            self.x_embedding = nn.Embedding(
                n_vertex + 2, x_emb_dim, padding_idx=n_vertex, device=device
            )

        # down blocks (✅ ModuleList로 등록)
        in_out = list(zip(dims, dims[1:]))
        self.down_blocks = nn.ModuleList([
            UnetBlock(in_dim, time_dim, out_dim, device, down_up="down", last=(i == len(in_out) - 1))
            for i, (in_dim, out_dim) in enumerate(in_out)
        ])

        mid_dim = dims[-1]

        # middle: 더 가볍게 (ResBlock 1개 + optional attn)
        self.mid_block = XTResBlock(mid_dim, time_dim, mid_dim, device)
        self.mid_attn = Residual(LinearAttention(mid_dim, device)) if use_mid_attn else None

        # pooling + classifier: 작게
        self.pool = nn.AdaptiveAvgPool1d(1)
        cls_hidden = mid_dim if cls_hidden is None else cls_hidden
        if cls_hidden == mid_dim:
            # 가장 가벼운 버전: 1-layer
            self.classifier = nn.Linear(mid_dim, 1, device=device)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(mid_dim, cls_hidden, device=device),
                nn.SiLU(),
                nn.Linear(cls_hidden, 1, device=device),
            )

    def forward(self, xt_padded, lengths, t, adj_matrix=None):
        # t embedding
        t = self.time_mlp(t)

        # xt -> token emb
        if xt_padded.dtype in (torch.float16, torch.float32, torch.float64):
            # xt_padded: [B,H,V] (soft one-hot/prob)
            # E: [Vocab, D]
            E = self.x_embedding.weight
            x = torch.einsum("bhv,vd->bhd", xt_padded, E)
        else:
            # xt_padded: [B,H]
            x = self.x_embedding(xt_padded)

        # down path
        for i, down_block in enumerate(self.down_blocks):
            x, _ = down_block(x, lengths if i == 0 else None, t)

        # mid
        x = self.mid_block(x, t)
        if self.mid_attn is not None:
            x = self.mid_attn(x, None)

        # global pooling + cls
        x = x.transpose(1, 2)          # [B, D, H]
        x = self.pool(x).squeeze(-1)   # [B, D]
        x = self.classifier(x).squeeze(-1)  # [B]
        return x
