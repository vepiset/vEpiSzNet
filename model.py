import math
import sys

sys.path.append('.')
import timm
import torch
import torch.nn as nn
from pytorchvideo.models.x3d import create_x3d
from timm.models.vision_transformer import VisionTransformer, Block, Attention


import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from lib.core.base_trainer.rope_embedding import RotaryEmbedding
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

torch._dynamo.config.cache_size_limit = 16
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_conv = nn.Sequential(nn.Conv3d(in_channels=3,
                                                out_channels=3,
                                                kernel_size=(3, 3, 3),
                                                stride=[2, 2, 2],
                                                padding=[1, 1, 1]
                                                ),
                                      nn.BatchNorm3d(3))
        model_name = "x3d_m"
        self.net = torch.hub.load('facebookresearch/pytorchvideo',
                                  model_name,
                                  pretrained=True, )

        # modified
        # self.net.blocks[0].conv.conv_xy.stride=[2,1,1]
        self.net.blocks[5] = nn.Identity()
        # we use it as seq
        # self.net.blocks[5].pool.pool = nn.AdaptiveMaxPool3d(1)
        # self.net.blocks[5].dropout = nn.Identity()
        # self.net.blocks[5].proj = nn.Identity()
        # self.net.blocks[5].activation = nn.Identity()
        # self.net.blocks[5].output_pool = nn.Identity()
        self.post_conv = nn.Sequential(nn.Conv3d(in_channels=192,
                                                 out_channels=512,
                                                 kernel_size=3,
                                                 padding=0,
                                                 stride=3, ))
        # print(self.net)

    def forward(self, x):
        x = x / 255.
        # use a pre conv to reduce cost
        # x = self.pre_conv(x)
        x = self.net(x)
        x = self.post_conv(x)
        x = torch.flatten(x, start_dim=2)
        x = x.transpose(1, 2)

        return x


class EEGModel(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.model = timm.create_model('convnextv2_nano.fcmae_ft_in22k_in1k_384',
                                       pretrained=True,
                                       in_chans=1,
                                       features_only=True
                                       )

    def forward(self, x, image=None, targets=None):
        bs = x.size(0)

        # x shape is (bs, 2, 19,1000)
        x = x.view(bs, 1, 19, 2000) / 32.
        reshaped_tensor = x.view(bs, 1, 19, 250, 8)
        reshaped_and_permuted_tensor = reshaped_tensor.permute(0, 1, 2, 4, 3)
        x = reshaped_and_permuted_tensor.reshape(bs, 1, 19 * 8, 250)
        # 160, 256, then all info will be kept
        x = F.pad(x, (0, 6, 0, 8), mode='constant')

        xs = self.model(x)

        x = xs[-1]
        x = torch.flatten(x, start_dim=2)
        x = x.transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, hidden_size=512, ratio=4, bias: bool = False, proj_drop=0.):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * ratio
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = nn.SiLU()
        self.proj_drop = nn.Dropout(proj_drop)
        self.init_weights()

    def forward(self, hidden_state):
        gate_output = self.act_fn(self.gate_proj(hidden_state))
        up_output = self.up_proj(hidden_state)
        intermediate = gate_output * up_output
        output = self.down_proj(intermediate)
        return self.proj_drop(output)  # 在最终输出前应用dropout

    def init_weights(self):
        """初始化MLP权重 - 使用类似GPT的初始化策略"""
        std = 0.02  # 标准的transformer初始化标准差

        # 使用正态分布初始化
        nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.trunc_normal_(self.up_proj.weight, mean=0.0, std=std)

        # down_proj使用较小的标准差，因为它是residual path的一部分
        nn.init.trunc_normal_(self.down_proj.weight, mean=0.0, std=std / math.sqrt(2))

        # 如果有bias，初始化为0
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

        self.init_weights()

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

    def init_weights(self):
        """初始化RMSNorm权重"""
        nn.init.ones_(self.weight)  # 初始化为1


class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # RoPE
        self.rope = RotaryEmbedding(dim=self.head_dim)

        self.init_weights()

        self.flash = True

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]
        with torch.cuda.amp.autocast(enabled=False):
            # 应用RoPE到q和k
            q = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 计算注意力
        # 计算注意力
        if self.flash:
            # Flash Attention需要的输入格式: [B, N, num_heads, head_dim]
            q = q.transpose(1, 2)  # [B, N, num_heads, head_dim]
            k = k.transpose(1, 2)  # [B, N, num_heads, head_dim]
            v = v.transpose(1, 2)  # [B, N, num_heads, head_dim]

            # 使用Flash Attention
            x = flash_attn_func(q, k, v)  # 输出: [B, N, num_heads, head_dim]

            # 重塑回原始维度
            x = x.reshape(B, N, C)
        else:
            # 标准注意力机制
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def init_weights(self):
        """初始化注意力权重"""
        std = 0.02

        # QKV权重初始化
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)

        # 输出投影权重初始化 - 使用较小的std因为是residual path
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=std / math.sqrt(2))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class MergeBlock(nn.Module):
    def __init__(self,
                 hidden_size=512,
                 num_heads=8,
                 mlp_ratio=4,
                 bias=False,
                 eps=1e-6,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.):
        super().__init__()
        self.hidden_size = hidden_size

        # Pre-attention normalization
        self.attention_norm = RMSNorm(hidden_size, eps=eps)

        # Multi-head self-attention
        self.attention = RoPEAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Pre-MLP normalization
        self.ffn_norm = RMSNorm(hidden_size, eps=eps)

        # Feed-forward network (MLP)
        self.mlp = MLP(
            hidden_size=hidden_size,
            ratio=mlp_ratio,
            bias=bias,
            proj_drop=proj_drop,
        )

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection
        normed_hidden_states = self.attention_norm(hidden_states)
        attn_output = self.attention(normed_hidden_states)
        hidden_states = hidden_states + self.drop_path1(attn_output)

        # MLP with residual connection
        normed_hidden_states = self.ffn_norm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = hidden_states + self.drop_path2(mlp_output)

        return hidden_states


class Net(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.eeg_model = EEGModel()
        self.image_model = ImageModel()

        image_feature_dim = 512
        eeg_feature_dim = 640
        embedding_dim = 384

        self.image_feature_proj = nn.Linear(image_feature_dim, out_features=embedding_dim, bias=True)
        self.eeg_feature_proj = nn.Linear(eeg_feature_dim, out_features=embedding_dim, bias=True)


        depth=6
        max_drop_path=0.2
        dpr = [x.item() for x in torch.linspace(0, max_drop_path, depth)]

        layers = []
        for i in range(depth):
            layer = MergeBlock(
                hidden_size=embedding_dim,
                num_heads=6,
                mlp_ratio=4,
                drop_path=dpr[i]
            )
            layers.append(layer)
        self.transformer = nn.ModuleList(layers)

        # 75+42+1
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.norm = RMSNorm(embedding_dim)
        self.dropout = nn.Dropout(p=0.1)

        self.task1 = nn.Linear(embedding_dim, out_features=2, bias=True)
        self.task2 = nn.Linear(embedding_dim, out_features=3, bias=True)
        self.task3 = nn.Linear(embedding_dim, out_features=10, bias=True)

        self.init_weights()

    def init_weights(self):
        """初始化整个网络的权重"""
        std = 0.02

        # 初始化class embedding
        nn.init.trunc_normal_(self.class_embedding, mean=0.0, std=std)

        # 初始化特征投影层
        nn.init.trunc_normal_(self.image_feature_proj.weight, mean=0.0, std=std)
        nn.init.zeros_(self.image_feature_proj.bias)

        nn.init.trunc_normal_(self.eeg_feature_proj.weight, mean=0.0, std=std)
        nn.init.zeros_(self.eeg_feature_proj.bias)

        # 初始化最终的分类头
        nn.init.trunc_normal_(self.task1.weight, mean=0.0, std=std)
        nn.init.zeros_(self.task1.bias)
        nn.init.trunc_normal_(self.task2.weight, mean=0.0, std=std)
        nn.init.zeros_(self.task2.bias)
        nn.init.trunc_normal_(self.task3.weight, mean=0.0, std=std)
        nn.init.zeros_(self.task3.bias)

    def _add_cls_token(self, x):
        batch_size = x.size(0)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, x], dim=1)
        # 移除position embedding，因为RoPE会在attention中处理位置信息
        # embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

    @torch.compile
    def forward(self, waves, images=0, targets=None):
        bs = waves.size(0)

        x_eeg = self.eeg_model(waves)
        x_eeg = self.eeg_feature_proj(x_eeg)

        if images!=0:
            # project image and eeg feature
            x_image = self.image_model(images)
            x_image = self.image_feature_proj(x_image)

            # concat the feature in time dimension
            x = torch.cat([x_eeg , x_image],dim=1)
        else:
            x = x_eeg
        # position and cls token embedding
        x = self._add_cls_token(x)

        for merge_block in self.transformer:
            x = merge_block(x)

        x = self.norm(x)
        x = x[:, 0, :]

        x = self.dropout(x)

        logit1 = self.task1(x)
        logit2 = self.task2(x)
        logit3 = self.task3(x)
        output = self.head([logit1, logit2, logit3], targets)

        return output


    def criterion(self, preds, targets):

        valid_mask = (targets != -1)
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        if not valid_mask.any():
            # 如果没有有效目标，返回0损失或跳过这个batch
            return torch.tensor(0.0, requires_grad=True, device=preds.device)
        # ignore_index=-1 表示忽略掉 -1 这个类别的预测结果，因为我们只关心前两个类别的预测结果。
        loss = torch.nn.functional.cross_entropy(preds, targets, ignore_index=-1)
        # loss = F.nll_loss(torch.log(preds), targets)
        return loss

    def head(self, logits, targets):

        [logit1, logit2, logit3] = logits

        if targets is not None:
            target1 = targets[0]
            target2 = targets[1]
            target3 = targets[2]
            loss = self.criterion(logit1, target1)  + \
            self.criterion(logit2, target2) +\
            self.criterion(logit3, target3)


        else:
            target1 = None
            target2 = None
            target3 = None
            loss = None

        prediction1 = torch.softmax(logit1, -1)
        prediction2 = torch.softmax(logit2, -1)
        prediction3 = torch.softmax(logit3, -1)

        return_dict = {
            'loss': loss,
            'prediction1': prediction1,
            'prediction2': prediction2,
            'prediction3': prediction3,
            'target1': target1,
            'target2': target2,
            'target3': target3,
        }

        return return_dict


if __name__ == '__main__':
    import torch
    import torchvision
    from thop import profile

    from thop import clever_format

    dummy_waves = torch.randn(1, 19, 2000, device='cpu')
    dummy_images = torch.randn(1, 3, 30, 540, 960, device='cpu')
    model = Net()

    # x = model( dummy_waves)
    macs, params = profile(model, inputs=[dummy_waves, dummy_images])
    macs, params = clever_format([macs, params], "%.3f")

    print(macs, params)
