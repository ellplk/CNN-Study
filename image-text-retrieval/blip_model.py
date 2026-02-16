"""
BLIP模型从零实现
包含: Vision Encoder, Text Encoder/Decoder, 多模态融合
支持: 图像描述生成, 视觉问答, 图文检索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """将图像分割成patch并进行嵌入"""
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(
        self, 
        embed_dim: int = 768, 
        num_heads: int = 12, 
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor = None, 
        value: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = query
        
        B, N, C = query.shape
        
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        
        return x


class MLP(nn.Module):
    """前馈神经网络"""
    
    def __init__(
        self, 
        embed_dim: int = 768, 
        hidden_dim: int = 3072, 
        dropout: float = 0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(
        self, 
        embed_dim: int = 768, 
        num_heads: int = 12, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask=attention_mask))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """视觉编码器 (ViT)"""
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        embed_dim: int = 768, 
        depth: int = 12, 
        num_heads: int = 12, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        cls_output = x[:, 0]
        patch_output = x[:, 1:]
        
        return cls_output, patch_output


class TextEmbedding(nn.Module):
    """文本嵌入层"""
    
    def __init__(
        self, 
        vocab_size: int = 30524, 
        embed_dim: int = 768, 
        max_position: int = 512, 
        dropout: float = 0.0
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_position, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.position_embed.weight, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        position_ids: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        token_embeds = self.token_embed(input_ids)
        position_embeds = self.position_embed(position_ids)
        
        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)
        
        return embeddings


class TextEncoder(nn.Module):
    """文本编码器 (BERT-style)"""
    
    def __init__(
        self, 
        vocab_size: int = 30524, 
        embed_dim: int = 768, 
        depth: int = 12, 
        num_heads: int = 12, 
        mlp_ratio: float = 4.0, 
        max_position: int = 512, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.embeddings = TextEmbedding(vocab_size, embed_dim, max_position, dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.embeddings(input_ids)
        
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None
        
        for block in self.blocks:
            x = block(x, extended_mask)
        
        x = self.norm(x)
        
        return x


class CrossAttentionBlock(nn.Module):
    """交叉注意力块 (用于解码器)"""
    
    def __init__(
        self, 
        embed_dim: int = 768, 
        num_heads: int = 12, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor = None,
        cross_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), attention_mask=self_attention_mask))
        x = x + self.dropout(self.cross_attn(
            self.norm2(x), 
            key=encoder_hidden_states, 
            value=encoder_hidden_states,
            attention_mask=cross_attention_mask
        ))
        x = x + self.mlp(self.norm3(x))
        return x


class TextDecoder(nn.Module):
    """文本解码器 (用于生成任务)"""
    
    def __init__(
        self, 
        vocab_size: int = 30524, 
        embed_dim: int = 768, 
        depth: int = 12, 
        num_heads: int = 12, 
        mlp_ratio: float = 4.0, 
        max_position: int = 512, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.embeddings = TextEmbedding(vocab_size, embed_dim, max_position, dropout)
        
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def _generate_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = input_ids.size(1)
        
        x = self.embeddings(input_ids)
        
        causal_mask = self._generate_causal_mask(seq_length, input_ids.device)
        
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
            causal_mask = causal_mask + extended_mask
        else:
            extended_mask = None
        
        for block in self.blocks:
            x = block(x, encoder_hidden_states, causal_mask, None)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


class BLIPModel(nn.Module):
    """完整的BLIP模型"""
    
    def __init__(
        self, 
        vocab_size: int = 30524, 
        img_size: int = 224, 
        patch_size: int = 16, 
        embed_dim: int = 768, 
        vision_depth: int = 12, 
        text_depth: int = 12, 
        num_heads: int = 12, 
        mlp_ratio: float = 4.0, 
        max_position: int = 512, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.vision_encoder = VisionEncoder(
            img_size, patch_size, 3, embed_dim, vision_depth, 
            num_heads, mlp_ratio, dropout
        )
        
        self.text_encoder = TextEncoder(
            vocab_size, embed_dim, text_depth, num_heads, 
            mlp_ratio, max_position, dropout
        )
        
        self.text_decoder = TextDecoder(
            vocab_size, embed_dim, text_depth, num_heads, 
            mlp_ratio, max_position, dropout
        )
        
        self.vision_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        
        self.itm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2)
        )
        
        self.temp = nn.Parameter(torch.ones(1) * 0.07)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        cls_output, _ = self.vision_encoder(image)
        image_embed = self.vision_proj(cls_output)
        image_embed = F.normalize(image_embed, dim=-1)
        return image_embed
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """编码文本"""
        text_output = self.text_encoder(input_ids, attention_mask)
        text_embed = self.text_proj(text_output[:, 0])
        text_embed = F.normalize(text_embed, dim=-1)
        return text_embed
    
    def compute_similarity(
        self, 
        image: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """计算图文相似度"""
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(input_ids, attention_mask)
        
        similarity = image_embed @ text_embed.T / self.temp.exp()
        return similarity
    
    def image_text_matching(
        self, 
        image: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """图文匹配 (ITM)"""
        _, patch_output = self.vision_encoder(image)
        text_output = self.text_encoder(input_ids, attention_mask)
        
        combined = torch.cat([patch_output, text_output], dim=1)
        cls_token = combined.mean(dim=1)
        
        itm_output = self.itm_head(cls_token)
        return itm_output
    
    def generate_caption(
        self, 
        image: torch.Tensor, 
        max_length: int = 50, 
        bos_token_id: int = 101, 
        eos_token_id: int = 102
    ) -> torch.Tensor:
        """生成图像描述"""
        _, patch_output = self.vision_encoder(image)
        
        batch_size = image.size(0)
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=image.device)
        
        for _ in range(max_length):
            logits = self.text_decoder(generated, patch_output)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == eos_token_id).all():
                break
        
        return generated
    
    def answer_question(
        self, 
        image: torch.Tensor, 
        question_ids: torch.Tensor, 
        max_length: int = 30, 
        bos_token_id: int = 101, 
        eos_token_id: int = 102
    ) -> torch.Tensor:
        """视觉问答"""
        _, patch_output = self.vision_encoder(image)
        
        question_output = self.text_encoder(question_ids)
        
        encoder_hidden_states = torch.cat([patch_output, question_output], dim=1)
        
        batch_size = image.size(0)
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=image.device)
        
        for _ in range(max_length):
            logits = self.text_decoder(generated, encoder_hidden_states)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == eos_token_id).all():
                break
        
        return generated


def create_blip_base():
    """创建BLIP-Base模型"""
    return BLIPModel(
        vocab_size=30524,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        vision_depth=12,
        text_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        max_position=512,
        dropout=0.0
    )


def create_blip_small():
    """创建BLIP-Small模型"""
    return BLIPModel(
        vocab_size=30524,
        img_size=224,
        patch_size=16,
        embed_dim=512,
        vision_depth=8,
        text_depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        max_position=512,
        dropout=0.0
    )


if __name__ == "__main__":
    print("=" * 60)
    print("BLIP模型架构测试")
    print("=" * 60)
    
    model = create_blip_base()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print(f"\n模型结构:")
    print(f"  Vision Encoder: ViT-B/16")
    print(f"  Text Encoder: BERT-base")
    print(f"  Embedding维度: 768")
    print(f"  注意力头数: 12")
    print(f"  Transformer层数: 12")
    
    print("\n测试前向传播...")
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 30524, (batch_size, 32))
    attention_mask = torch.ones(batch_size, 32)
    
    with torch.no_grad():
        image_embed = model.encode_image(image)
        text_embed = model.encode_text(input_ids, attention_mask)
        
        print(f"\n输出形状:")
        print(f"  图像特征: {image_embed.shape}")
        print(f"  文本特征: {text_embed.shape}")
        
        similarity = model.compute_similarity(image, input_ids, attention_mask)
        print(f"  相似度矩阵: {similarity.shape}")
        
        itm_output = model.image_text_matching(image, input_ids, attention_mask)
        print(f"  ITM输出: {itm_output.shape}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
