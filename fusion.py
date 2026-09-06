# define the fusion module for combining the three modalities

import torch
import torch.nn as nn

try:
    from . import Pose_RS
except ImportError:
    import Pose_RS

class VideoADLFeatureExtractor(nn.Module):
    """
    Expected input:
        [B, 13, 192, 6, 6]

    Output:
        [B, 13, 25, 6, 6]
    """

    def __init__(
        self,
        in_channels: int = 192,
        out_channels: int = 25,
        temporal_kernel_size: int = 5,
    ):
        super().__init__()

        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size should be odd.")

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Spatial convolution:
        # extracts per-frame geometric features
        # and compresses channels 192 -> 25.
        self.conv_xy = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            bias=False,
        )

        # Temporal depthwise convolution:
        # models temporal changes independently for each channel.
        self.conv_t = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            stride=(1, 1, 1),
            padding=(temporal_kernel_size // 2, 0, 0),
            groups=out_channels,
            bias=False,
        )

        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected [B,T,C,H,W] or [B,C,T,H,W], got {tuple(x.shape)}"
            )

        input_is_btchw = x.shape[2] == self.in_channels
        if input_is_btchw:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        elif x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected channel dim {self.in_channels} at axis 1 or 2, got {tuple(x.shape)}"
            )

        x = self.conv_xy(x)
        # [B,192,13,6,6] -> [B,25,13,6,6]

        x = self.conv_t(x)
        # [B,25,13,6,6] -> [B,25,13,6,6]

        x = self.bn(x)
        x = self.relu(x)

        return x.permute(0, 2, 1, 3, 4).contiguous()


class ADLFeatureReducer(nn.Module):
    """Legacy reducer: flatten the whole fused map and project with one big FC.

    Kept for backward compatibility and as an ablation baseline. The final
    linear layer alone holds ~12.9M weights, which is where most of the old
    parameter budget (and most of the subject overfitting) came from.
    """

    def __init__(
        self,
        in_channels=50,
        spatial_channels=50,
        temporal_channels=50,
        output_dim=512,
        flattened_dim=25200,
        dropout=0.2,
    ):
        super().__init__()

        self.spatial_conv = nn.Conv2d(
            in_channels,
            spatial_channels,
            kernel_size=3,
            padding=1
        )

        self.temporal_conv = nn.Conv1d(
            spatial_channels,
            temporal_channels,
            kernel_size=3,
            padding=1
        )

        self.temporal_bn = nn.BatchNorm1d(temporal_channels)
        self.relu = nn.ReLU(inplace=True)
        self.flattened_dim = flattened_dim
        self.fc = nn.Linear(flattened_dim, output_dim)
        self.fc_bn = nn.BatchNorm1d(output_dim)
        self.fc_act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")

        B, T, C, H, W = x.shape

        # 空间卷积
        x = x.reshape(B * T, C, H, W)
        x = self.spatial_conv(x)

        _, C2, H2, W2 = x.shape
        x = x.reshape(B, T, C2, H2, W2)

        # 调整为 Conv1d 需要的 [batch, channel, time]
        x = x.permute(0, 3, 4, 2, 1)
        x = x.reshape(B * H2 * W2, C2, T)

        # 时间卷积
        x = self.relu(self.temporal_bn(self.temporal_conv(x)))

        x = x.reshape(B, -1)
        if x.shape[1] != self.flattened_dim:
            raise ValueError(
                f"Expected flattened ADL feature length {self.flattened_dim}, got {x.shape[1]}"
            )

        x = self.fc(x)
        x = self.fc_bn(x)
        x = self.fc_act(x)
        x = self.dropout(x)

        return x


class TokenAttentionADLFeatureReducer(nn.Module):
    """Reduce fused ADL maps with pooled temporal tokens and self-attention.

    This avoids the old 25,200 -> 512 flatten projection, which can memorize
    train-subject spatial details too easily. Each temporal slice becomes one
    token after lightweight spatial encoding and average pooling.
    """

    def __init__(
        self,
        in_channels=50,
        token_dim=256,
        output_dim=512,
        num_layers=2,
        num_heads=4,
        max_tokens=16,
        dropout=0.4,
    ):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")

        self.max_tokens = max_tokens
        self.frame_encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=token_dim,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(token_dim),
            nn.GELU(),
            nn.Conv3d(
                in_channels=token_dim,
                out_channels=token_dim,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                groups=token_dim,
                bias=False,
            ),
            nn.BatchNorm3d(token_dim),
            nn.GELU(),
        )
        self.token_norm = nn.LayerNorm(token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_tokens + 1, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")

        batch_size, frames, _channels, _height, _width = x.shape
        if frames > self.max_tokens:
            raise ValueError(f"Expected at most {self.max_tokens} temporal tokens, got {frames}")

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.frame_encoder(x)
        tokens = x.mean(dim=(-1, -2)).transpose(1, 2).contiguous()
        tokens = self.token_norm(tokens)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : frames + 1]
        tokens = self.encoder(tokens)
        return self.projection(tokens[:, 0])


class SpatialTokenAttentionReducer(nn.Module):
    """Attention reducer that keeps the spatial layout of the RS maps.

    The plain attention reducer averages each frame's 6x6 grid into a single
    token, which throws away exactly the information the RS maps encode: where
    on the grid the joints and objects sit. That cost it the closed-set edge
    against the flatten baseline. Three changes here:

      1. each frame becomes a 3x3 grid of tokens instead of one, so quadrant-
         level position survives into the transformer;
      2. every 2x2 window is pooled with a learned per-cell attention instead
         of a mean, so a single active RS cell is not diluted by three idle ones;
      3. positional embeddings are factorised into temporal x spatial parts.
    """

    def __init__(
        self,
        in_channels=50,
        token_dim=256,
        output_dim=512,
        num_layers=1,
        num_heads=4,
        max_tokens=16,
        dropout=0.3,
        window=2,
    ):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")

        self.max_tokens = max_tokens
        self.window = window
        self.frame_encoder = nn.Sequential(
            nn.Conv3d(in_channels, token_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(token_dim),
            nn.GELU(),
            nn.Conv3d(token_dim, token_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=token_dim, bias=False),
            nn.BatchNorm3d(token_dim),
            nn.GELU(),
        )
        # One logit per grid cell; softmax runs inside each pooling window.
        self.cell_score = nn.Conv3d(token_dim, 1, kernel_size=1)

        self.token_norm = nn.LayerNorm(token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        grid_tokens = (6 // window) ** 2
        self.spatial_embedding = nn.Parameter(torch.zeros(1, 1, grid_tokens, token_dim))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, max_tokens, 1, token_dim))
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        for parameter in (self.cls_token, self.spatial_embedding, self.temporal_embedding, self.cls_embedding):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")

        batch_size, frames, _channels, height, width = x.shape
        if frames > self.max_tokens:
            raise ValueError(f"Expected at most {self.max_tokens} temporal tokens, got {frames}")

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.frame_encoder(x)                                 # [B,D,T,6,6]

        # Learned pooling: softmax the cell scores within each 2x2 window and
        # take the weighted sum, so the pooled token follows the active cell.
        w = self.window
        scores = self.cell_score(x)                               # [B,1,T,6,6]
        dim = x.shape[1]
        # [B,D,T,6,6] -> [B,D,T,3,3,2,2]: grid dims first, window cells last.
        x = x.reshape(batch_size, dim, frames, height // w, w, width // w, w)
        x = x.permute(0, 1, 2, 3, 5, 4, 6).flatten(-2, -1)
        scores = scores.reshape(batch_size, 1, frames, height // w, w, width // w, w)
        scores = scores.permute(0, 1, 2, 3, 5, 4, 6).flatten(-2, -1)
        weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        tokens = (x * weights).sum(dim=-1)                        # [B,D,T,3,3]

        tokens = tokens.flatten(-2, -1).permute(0, 2, 3, 1)       # [B,T,9,D]
        tokens = self.token_norm(tokens)
        tokens = tokens + self.temporal_embedding[:, :frames] + self.spatial_embedding
        tokens = tokens.flatten(1, 2)                             # [B,T*9,D]

        cls = self.cls_token.expand(batch_size, -1, -1) + self.cls_embedding
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.encoder(tokens)
        return self.projection(tokens[:, 0])


class GapADLFeatureReducer(nn.Module):
    """Spatial GAP + temporal GAP, then a small MLP to the ADL embedding.

    The cheapest of the three reducers. It keeps the same pooling front-end as
    the attention version but drops the transformer, so it isolates how much of
    the gain comes from pooling and how much from temporal attention.
    """

    def __init__(self, in_channels=50, hidden_dim=256, output_dim=512, dropout=0.2):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.frame_encoder(x)
        # Pool over time and space in one step: [B,C,T,H,W] -> [B,C]
        x = x.mean(dim=(-3, -2, -1))
        return self.projection(x)


class ObjectCrossAttention(nn.Module):
    """Model human-object interaction with cross attention.

    Key/Value are one token per object category, pooled from that category's RS
    map. Queries come from the video-pose maps, and two choices matter:

    ``spatial_query`` puts one query on every grid cell instead of one per
    frame, so the answer stays spatial - cell (i,j) of frame t gets the object
    context relevant *at that location*. Pooling the query per frame throws away
    where the object was, which is the whole point of an RS map.

    ``fuse="concat"`` appends the context to the channels instead of adding it
    through a learned gate. The first version used a zero-initialised gate, and
    training drove it to ~0.04: multiplied by the output projection that is an
    injection of ~0.002, i.e. the object stream had switched itself off. Concat
    removes that escape hatch. ``fuse="residual"`` keeps the old behaviour so
    earlier checkpoints still load.
    """

    def __init__(
        self,
        feature_channels=50,
        num_objects=50,
        grid_size=6,
        embed_dim=128,
        num_heads=4,
        dropout=0.1,
        fuse="concat",
        spatial_query=True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if fuse not in {"concat", "residual"}:
            raise ValueError(f"fuse must be 'concat' or 'residual', got {fuse!r}")

        self.fuse = fuse
        self.spatial_query = spatial_query
        self.query_proj = nn.Linear(feature_channels, embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.object_proj = nn.Linear(grid_size * grid_size, embed_dim)
        self.object_norm = nn.LayerNorm(embed_dim)
        # Channel index of the RS map is the object category, so give each
        # category its own learned identity token.
        self.object_embedding = nn.Parameter(torch.zeros(num_objects, embed_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(embed_dim, feature_channels)
        if fuse == "residual":
            self.gate = nn.Parameter(torch.zeros(1))
        # Kept for interpretability: which object the queries attended to.
        self.last_attention = None

        nn.init.trunc_normal_(self.object_embedding, std=0.02)

    def forward(self, vp_feature, object_map):
        # vp_feature: [B,T,C,6,6], object_map: [B,O,6,6]
        if object_map.ndim != 4:
            raise ValueError(f"Expected object_map [B,O,H,W], got {tuple(object_map.shape)}")

        batch, frames, channels, height, width = vp_feature.shape
        if self.spatial_query:
            # [B,T,C,H,W] -> [B, T*H*W, C]: one query per frame and grid cell.
            queries = vp_feature.permute(0, 1, 3, 4, 2).reshape(batch, -1, channels)
        else:
            queries = vp_feature.mean(dim=(-1, -2))
        queries = self.query_norm(self.query_proj(queries))

        keys = self.object_proj(object_map.flatten(2)) + self.object_embedding
        keys = self.object_norm(keys)

        # An all-zero channel means the detector did not see that category.
        absent = object_map.amax(dim=(-1, -2)) <= 0
        # Masking every key would make the softmax undefined, so a sample with no
        # detection at all attends freely and the zero-valued keys carry nothing.
        absent = absent & ~absent.all(dim=1, keepdim=True)

        context, weights = self.attention(queries, keys, keys, key_padding_mask=absent)
        self.last_attention = weights.detach()
        context = self.out_proj(context)

        if self.spatial_query:
            context = context.reshape(batch, frames, height, width, channels)
            context = context.permute(0, 1, 4, 2, 3)
        else:
            context = context[..., None, None].expand(-1, -1, -1, height, width)

        if self.fuse == "residual":
            return vp_feature + self.gate * context
        return torch.cat([vp_feature, context], dim=2)


def build_reducer(
    reducer_type,
    in_channels,
    num_frames,
    dropout,
    attention_dim,
    attention_heads,
    attention_layers,
):
    name = str(reducer_type).lower()
    if name in {"flatten_fc", "fc", "legacy"}:
        return ADLFeatureReducer(
            in_channels=in_channels,
            spatial_channels=50,
            temporal_channels=50,
            output_dim=512,
            flattened_dim=50 * 6 * 6 * num_frames,
            dropout=dropout,
        )
    if name in {"attention_pool", "token_attention", "attention"}:
        return TokenAttentionADLFeatureReducer(
            in_channels=in_channels,
            token_dim=attention_dim,
            output_dim=512,
            num_layers=attention_layers,
            num_heads=attention_heads,
            max_tokens=16,
            dropout=dropout,
        )
    if name in {"spatial_attention", "spatial_tokens"}:
        return SpatialTokenAttentionReducer(
            in_channels=in_channels,
            token_dim=attention_dim,
            output_dim=512,
            num_layers=attention_layers,
            num_heads=attention_heads,
            max_tokens=16,
            dropout=dropout,
        )
    if name in {"gap", "gap_pool"}:
        return GapADLFeatureReducer(
            in_channels=in_channels,
            hidden_dim=attention_dim,
            output_dim=512,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported reducer_type: {reducer_type}")


class PoseADLFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, grid_size=6):
        super().__init__()
        self.person_attn = nn.Sequential(
            nn.Linear(in_channels, 1)
        )

        self.to_grid = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, grid_size * grid_size),
        )

        self.grid_size = grid_size

    def forward(self, x):
        # x: [B, M, C, T, V] = [B,2,64,13,25]
        B, M, C, T, V = x.shape

        # [B,M,C,T,V] -> [B,T,V,M,C]
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        # 对 person 做 attention
        # attn_logits: [B,T,V,M,1]
        attn_logits = self.person_attn(x)
        attn = torch.softmax(attn_logits, dim=3)

        # 融合 person: [B,T,V,C]
        x = (x * attn).sum(dim=3)

        # 每个 time/joint 的 64 维特征 -> 36
        # [B,T,V,C] -> [B,T,V,36]
        x = self.to_grid(x)

        # [B,T,V,36] -> [B,T,V,6,6]
        x = x.view(B, T, V, self.grid_size, self.grid_size)

        return x

class TriModalFusion(nn.Module):
    """
    TriModalFusion

    inputs：
        Video_feature: [B, 13, 192, 6, 6]
        Pose_feature: [B, 2, 64, 13, V]
        Object_feature: [B, 1, 50, 6, 6] or [B, 50, 6, 6]
        Joint_location: [B, 13, V, 2]
        V is 25 for Kinect skeletons, 17 for RTMPose COCO-17.

    hidden outputs：
        Video_feature: [B, 13, video_channels, 6, 6]
        Pose_feature: [B, 13, V, 6, 6]
        Object_feature: [B, 1, 50, 6, 6]
        Fused_feature: [B, 13, 50, 6, 6]   (14 frames with temporal_concat,
                                            100 channels with broadcast)

    outputs：
        ADL_embedding: [B, 512]
    """
    def __init__(
        self,
        reducer_dropout: float = 0.2,
        reducer_type: str = "attention_pool",
        attention_dim: int = 256,
        attention_heads: int = 4,
        attention_layers: int = 2,
        object_fusion: str = "cross_attention",
        object_attention_dim: int = 128,
        object_attention_heads: int = 4,
        object_dropout: float = 0.1,
        num_frames: int = 13,
        video_channels: int = 25,
        num_joints: int = 25,
    ):
        super().__init__()

        # The reducer expects video + pose to concatenate to 50 channels. With
        # Kinect-25 skeletons that is 25+25; with RTMPose COCO-17 it is 33+17.
        if video_channels + num_joints != 50:
            raise ValueError(
                f"video_channels + num_joints must be 50, got {video_channels}+{num_joints}"
            )
        self.num_joints = num_joints

        # 线性层用于对齐通道数
        self.video_raw_proj = VideoADLFeatureExtractor(in_channels=192, out_channels=video_channels)
        self.pose_raw_proj = PoseADLFeatureExtractor(in_channels=64, grid_size=6)

        self.object_fusion = str(object_fusion).lower()
        self.object_attention = None
        if self.object_fusion in {"cross_attention", "cross_attn", "cross_attention_residual"}:
            # The residual variant is the original design, kept so its
            # checkpoints still load; it lets the object stream gate itself off.
            residual = self.object_fusion == "cross_attention_residual"
            self.object_fusion = "cross_attention_residual" if residual else "cross_attention"
            self.object_attention = ObjectCrossAttention(
                feature_channels=50,
                num_objects=50,
                grid_size=6,
                embed_dim=object_attention_dim,
                num_heads=object_attention_heads,
                dropout=object_dropout,
                fuse="residual" if residual else "concat",
                spatial_query=not residual,
            )
            fused_channels = 50 if residual else 100
            fused_frames = num_frames
        elif self.object_fusion == "broadcast":
            fused_channels, fused_frames = 100, num_frames
        elif self.object_fusion in {"temporal_concat", "concat", "legacy"}:
            self.object_fusion = "temporal_concat"
            fused_channels, fused_frames = 50, num_frames + 1
        else:
            raise ValueError(f"Unsupported object_fusion: {object_fusion}")

        self.reducer = build_reducer(
            reducer_type,
            in_channels=fused_channels,
            num_frames=fused_frames,
            dropout=reducer_dropout,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
        )

    def fuse_object(self, vp_feature, object_proj):
        """Merge the object RS map into the video-pose stream.

        object_proj is [B,1,50,6,6]: 50 category channels, no real time axis.
        """

        if self.object_attention is not None:
            return self.object_attention(vp_feature, object_proj[:, 0])
        if self.object_fusion == "broadcast":
            spread = object_proj.expand(-1, vp_feature.shape[1], -1, -1, -1)
            return torch.cat([vp_feature, spread], dim=2)
        # temporal_concat: the original behaviour, object map as an extra step.
        return torch.cat([vp_feature, object_proj], dim=1)

    def forward(self, video_feature_raw, pose_feature_raw, object_feature_raw, joint_location_raw):
        # video_feature_raw: [B, 13, 192, 6, 6]
        # pose_feature_raw: [B, 2, 64, 13, 25]
        # object_feature_raw: [B, 1, 50, 6, 6]
        # joint_location_raw: [B, 13, 25, 2]

        video_proj = self.video_raw_proj(video_feature_raw)

        pose_proj = self.pose_raw_proj(pose_feature_raw)

        if joint_location_raw.ndim != 4 or joint_location_raw.shape[-1] != 2:
            raise ValueError(
                f"Expected joint_location_raw [B,13,V,2], got {tuple(joint_location_raw.shape)}"
            )
        if joint_location_raw.shape[2] != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joints, got {joint_location_raw.shape[2]}"
            )

        x_s = joint_location_raw[..., 0]
        y_s = joint_location_raw[..., 1]
        pose_proj = Pose_RS.get_RS_map(
            pose_proj, x_s, y_s
        )

        if object_feature_raw.ndim == 4:
            object_feature_raw = object_feature_raw.unsqueeze(1)
        if object_feature_raw.ndim != 5:
            raise ValueError(
                f"Expected object_feature_raw [B,1,50,6,6] or [B,50,6,6], got {tuple(object_feature_raw.shape)}"
            )

        # object feature now has shape [B, 1, 50, 6, 6]
        object_proj = torch.nan_to_num(
            object_feature_raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(min=0.0, max=10.0)
        if object_proj.shape[1:] != (1, 50, 6, 6):
            raise ValueError(
                f"Expected object feature [B,1,50,6,6], got {tuple(object_proj.shape)}"
            )


        # Video skip path + pose RS path:
        # concatenate video and pose along the channel dimension.
        vp_feature = torch.cat([video_proj, pose_proj], dim=2)
        # [B, 13, 50, 6, 6]

        fused_feature = self.fuse_object(vp_feature, object_proj)

        # ADL feature reducer -> [B,512]
        fused_feature = self.reducer(fused_feature)

        return fused_feature
