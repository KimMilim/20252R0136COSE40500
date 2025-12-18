"""
TODO 1: ViT 구현

PatchEmbedding: 이미지를 잘라서 벡터로 만드는 부분.

TransformerBlock: Attention과 MLP가 들어있는 인코더의 핵심 블록.

VisionTransformer: 전체를 조립하는 메인 클래스.


"""
import torch
import torch.nn as nn
import math

# =============================================================================
# TODO 2 (B) RoPE (Rotary Positional Embedding) [Helper Functions] 
# =============================================================================
def precompute_freqs_cis(dim, end, theta=10000.0):
    """
    미리 각도(freqs)의 복소수 값(cis = cos + i*sin)을 계산해두는 함수
    dim: Head Dimension
    end: Max Sequence Length
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device) 
    freqs = torch.outer(t, freqs).float()  # (end, dim//2)
    
    # 복소수 형태로 변환 (polar coordinates)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    xq, xk: (Batch, Seq_Len, n_heads, head_dim)
    freqs_cis: 미리 계산된 복소수 각도
    """
    # 실수(Real) 텐서를 복소수(Complex) 텐서로 변환 (view_as_complex)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 브로드캐스팅을 위해 차원 맞추기
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # 회전 적용 (복소수 곱셈: Rotation)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# =============================================================================
# TODO 2 (A) 2D Sinusoidal PE [Helper Functions]
# =============================================================================
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


# =============================================================================
# [Model Components] PatchEmbedding, MLP, Attention, Block
# =============================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) 
        x = x.flatten(2).transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=False, use_rope=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)


    def forward(self, x, freqs_cis=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # RoPE 적용을 쉽게 하기 위해 (3, B, N, heads, head_dim) 형태로 변환하지 않고
        # (B, N, heads, head_dim) 상태에서 처리할 준비를 함
        # 여기서는 전통적인 방식에 맞춰 Permute 후 분리
        qkv = qkv.permute(2, 0, 1, 3, 4) # (3, B, N, heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # 각각 (B, N, heads, head_dim)

        # TODO 2 (B) RoPE 적용
        # "Apply rotation to query/key vectors"
        if self.use_rope and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
        # ----------------------------

        # Attention 연산을 위해 (B, heads, N, head_dim)으로 전치
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.0, use_rope=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, x, freqs_cis=None):
        # Attention에 freqs_cis 전달
        x = x + self.attn(self.norm1(x), freqs_cis=freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# [Main Class] VisionTransformer
# =============================================================================
class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4., 
                 use_rope=False): 
        super().__init__()
        self.use_rope = use_rope
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches

        # 2. Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Positional Encoding Setup
        # TODO 2 (A) Baseline PE: use_rope=False일 때 사용
        # TODO 2 (B) RoPE: use_rope=True일 때 사용 (Baseline PE는 끔)
        
        if self.use_rope:
            print("[Info] RoPE is ENABLED. Standard Positional Embedding is DISABLED.")
            self.pos_embed = None
            
            # RoPE를 위한 주파수(freqs_cis) 미리 계산
            # Seq Len = num_patches + 1 (cls_token)
            head_dim = embed_dim // num_heads
            self.freqs_cis = precompute_freqs_cis(head_dim, num_patches + 1)
            
        else:
            print("[Info] Standard 2D Sinusoidal Positional Embedding is ENABLED.")
            self.freqs_cis = None
            
            # TODO 2 (A): 2D Sinusoidal PE 적용
            grid_size = img_size // patch_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
            )
            pos_embed_values = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)
            self.pos_embed.data.copy_(pos_embed_values.float().unsqueeze(0))

        self.pos_drop = nn.Dropout(p=0.1)

        # 4. Encoder Blocks (RoPE 사용 여부 전달)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, use_rope=use_rope) 
            for _ in range(depth)
        ])

        # 5. Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights_layer)

    def _init_weights_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x) 

        # Append CLS Token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # (A) Add Baseline PE (if exists)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        x = self.pos_drop(x)

        # (B) Prepare RoPE Freqs (if enabled)
        freqs_cis = None
        if self.use_rope:
            # 현재 배치(x)가 있는 디바이스로 freqs_cis를 이동
            freqs_cis = self.freqs_cis.to(x.device)
            # 입력 시퀀스 길이에 맞춰 슬라이싱 (보통 길이는 고정이지만 안전하게)
            freqs_cis = freqs_cis[:x.shape[1]]

        # Blocks Forward
        for blk in self.blocks:
            x = blk(x, freqs_cis=freqs_cis)

        # Head
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        
        return x