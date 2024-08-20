import jittor as jt
from jittor import init
from jittor import nn
from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
from .mha import MultiheadAttention,multi_head_attention_forward
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.relu2 = nn.ReLU()
        self.avgpool = (nn.Pool(stride, op='mean') if (stride > 1) else nn.Identity())
        self.conv3 = nn.Conv(planes, (planes * self.expansion), 1, bias=False)
        self.bn3 = nn.BatchNorm((planes * self.expansion))
        self.relu3 = nn.ReLU()
        self.downsample = None
        self.stride = stride
        if ((stride > 1) or (inplanes != (planes * Bottleneck.expansion))):
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.Pool(stride, op='mean')), ('0', nn.Conv(inplanes, (planes * self.expansion), 1, stride=1, bias=False)), ('1', nn.BatchNorm((planes * self.expansion)))]))

    def execute(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if (self.downsample is not None):
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int=None):
        super().__init__()
        self.positional_embedding = jt.array((jt.randn(((spacial_dim ** 2) + 1), embed_dim) / (embed_dim ** 0.5)))
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, (output_dim or embed_dim))
        self.num_heads = num_heads

    def execute(self, x):
        x = x.flatten(start_dim=2).permute((2, 0, 1))
        x = jt.contrib.concat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = (x + self.positional_embedding[:, None, :].to(x.dtype))
        (x, _) = multi_head_attention_forward(query=x[:1], key=x, value=x, embed_dim_to_check=x.shape[(- 1)], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=jt.contrib.concat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x.squeeze(0)

class ModifiedResNet(nn.Module):

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv(3, (width // 2), 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm((width // 2))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv((width // 2), (width // 2), 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm((width // 2))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv((width // 2), width, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm(width)
        self.relu3 = nn.ReLU()
        self.avgpool = nn.Pool(2, op='mean')
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer((width * 2), layers[1], stride=2)
        self.layer3 = self._make_layer((width * 4), layers[2], stride=2)
        self.layer4 = self._make_layer((width * 8), layers[3], stride=2)
        embed_dim = (width * 32)
        self.attnpool = AttentionPool2d((input_resolution // 32), embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = (planes * Bottleneck.expansion)
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def execute(self, x):

        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        x = x
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x

class LayerNorm(nn.LayerNorm):

    def execute(self, x):
        ret = super().execute(x)
        return ret


class QuickGELU(nn.Module):

    def execute(self, x):
        return x * jt.sigmoid(1.702 * x)

class MLP(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)

    def execute(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model, n_head, attn_mask):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = MLP(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False,
                         attn_mask=self.attn_mask)[0]

    def execute(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def execute(self, x):
        return self.resblocks(x)

class VisionTransformer(nn.Module):

    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width**-0.5
        self.class_embedding = scale * jt.randn((width))
        self.positional_embedding = scale * jt.randn(
            ((input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = scale * jt.randn((width, output_dim))

    def execute(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = jt.concat([
            self.class_embedding.to(x.dtype) + jt.zeros(
                (x.shape[0], 1, x.shape[-1]), dtype=x.dtype), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class CLIP(nn.Module):

    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: Union[(Tuple[(int, int, int, int)], int)], vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int, transformer_layers: int):
        super().__init__()
        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = ((vision_width * 32) // 64)
            self.visual = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width)
        else:
            vision_heads = (vision_width // 64)
            self.visual = VisionTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = jt.array(jt.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = jt.array(jt.empty(transformer_width, embed_dim))
        self.logit_scale = jt.array((jt.ones([]) * np.log((1 / 0.07))))
        self.initialize_parameters()

    def initialize_parameters(self):
        init.gauss_(self.token_embedding.weight, std=0.02)
        init.gauss_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if (self.visual.attnpool is not None):
                std = (self.visual.attnpool.c_proj.in_features ** (- 0.5))
                init.gauss_(self.visual.attnpool.q_proj.weight, std=std)
                init.gauss_(self.visual.attnpool.k_proj.weight, std=std)
                init.gauss_(self.visual.attnpool.v_proj.weight, std=std)
                init.gauss_(self.visual.attnpool.c_proj.weight, std=std)
            #for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
            #    for (name, param) in resnet_block.named_parameters():
                    #if name.endswith('bn3.weight'):
                        #nn.init.zeros_(param)
                    #    jt.zeros(param)
        proj_std = ((self.transformer.width ** (- 0.5)) * ((2 * self.transformer.layers) ** (- 0.5)))
        attn_std = (self.transformer.width ** (- 0.5))
        fc_std = ((2 * self.transformer.width) ** (- 0.5))
        for block in self.transformer.resblocks:
            init.gauss_(block.attn.in_proj_weight, std=attn_std)
            init.gauss_(block.attn.out_proj.weight, std=proj_std)
            init.gauss_(block.mlp.c_fc.weight, std=fc_std)
            init.gauss_(block.mlp.c_proj.weight, std=proj_std)
        if (self.text_projection is not None):
            init.gauss_(self.text_projection, std=(self.transformer.width ** (- 0.5)))

    def build_attention_mask(self):
        mask = jt.empty((self.context_length, self.context_length))
        mask.fill_(float("-inf"))
        mask = jt.triu(mask, 1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[jt.arange(x.shape[0]),
              text.argmax(dim=-1)[0]] @ self.text_projection
        return x

    def execute(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = (image_features / image_features.norm(dim=1, keepdim=True))
        text_features = (text_features / text_features.norm(dim=1, keepdim=True))
        logit_scale = self.logit_scale.exp()
        logits_per_image = ((logit_scale * image_features) @ text_features.t())
        logits_per_text = logits_per_image.t()
        return (logits_per_image, logits_per_text)


def build_model(state_dict: dict):
    vit = ('visual.proj' in state_dict)
    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([k for k in state_dict.keys() if (k.startswith('visual.') and k.endswith('.attn.in_proj_weight'))])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[(- 1)]
        grid_size = round(((state_dict['visual.positional_embedding'].shape[0] - 1) ** 0.5))
        image_resolution = (vision_patch_size * grid_size)
    else:
        counts: list = [len(set((k.split('.')[2] for k in state_dict if k.startswith(f'visual.layer{b}')))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round(((state_dict['visual.attnpool.positional_embedding'].shape[0] - 1) ** 0.5))
        vision_patch_size = None
        assert (((output_width ** 2) + 1) == state_dict['visual.attnpool.positional_embedding'].shape[0])
        image_resolution = (output_width * 32)
    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = (transformer_width // 64)
    transformer_layers = len(set((k.split('.')[2] for k in state_dict if k.startswith('transformer.resblocks'))))
    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
    for key in ['input_resolution', 'context_length', 'vocab_size']:
        if (key in state_dict):
            del state_dict[key]
    model.load_parameters(state_dict)
    return model.eval()
