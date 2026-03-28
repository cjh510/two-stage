from torch import Tensor, nn
from typing import Tuple, Type
import torch

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  #这行多了个qk_scale #0.125

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape #2 512 768
        q = self.wq(x[:, 0:int(N/2), ...]).reshape(B, int(N/2), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#2 12 256 64
        k = self.wk(x[:, (int(N/2)):, ...]).reshape(B, int(N/2), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x[:, (int(N/2)):, ...]).reshape(B, int(N/2), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, int(N/2), C) #变成了B/2 2 256 768
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_ori(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  #2,12,49,64  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale  #2,12,49,49 #mask 2,1,49,49

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class FusionAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
        """
        super().__init__()
        self.self_attn = Attention_ori(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_mask_to_image = CrossAttention(dim=embedding_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_mask = CrossAttention(dim=embedding_dim, num_heads=num_heads)


    def forward(self, img_emb: Tensor, mask_emb: Tensor) -> Tuple[ Tensor]:
        # Self attention block #最开始的时候 queries=query_pe
        #queries: Tensor, keys: Tensor
        queries = mask_emb
        attn_out = self.self_attn(queries)  #小图
        queries = attn_out
        # queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, mask attending to image embedding
        q = queries #1,5,256
        k = img_emb  # v是值，因此用keys？
        input_x = torch.cat((q, k), dim=1)  # 2 50 768
        attn_out = self.cross_attn_mask_to_image(input_x) #TODO 要不要mask呢 交叉的时候 先不用试试
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = img_emb
        k = queries
        input_x = torch.cat((q, k), dim=1)
        attn_out = self.cross_attn_image_to_mask(input_x)
        img_emb = img_emb + attn_out
        img_emb = self.norm4(img_emb)

        return img_emb