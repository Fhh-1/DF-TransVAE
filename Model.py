import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerVAEClassifier(nn.Module):
    # def __init__(self, input_dim=9, d_model=256, nhead=8, num_layers=4, latent_dim=64, num_heads=4):
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=4, latent_dim=256, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 特征投影
        self.feature_proj = nn.Linear(input_dim, d_model)  # 输入特征投影到 d_model

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # VAE 编码器部分
        # 修改点：输入维度变为 d_model + input_dim，因为拼接了 Transformer 输出和原始输入
        self.fc_mu = nn.Sequential(
            nn.Linear(d_model + input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )  # 均值映射
        self.fc_logvar = nn.Sequential(
            nn.Linear(d_model + input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )  # 方差映射

        # 交叉注意力模块
        self.cross_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)

        # 分类器（优化后，添加 BatchNorm 和更稳定的激活函数）
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )

        # 最后一层分类器
        self.final_layer = nn.Linear(32, 1)

        # 残差连接的线性层，确保输入是 latent_dim 维度
        self.residual_connection = nn.Linear(latent_dim, 32)  # 修正点: 确保输入与 mu 匹配

        # 归一化层
        self.norm = nn.LayerNorm(32)

        # 用于将 x_trans 转换为 latent_dim 维度的线性层
        self.x_trans_linear = nn.Linear(d_model, latent_dim)

    def reparameterize(self, mu, log_var):
        """重参数化技巧，从均值和方差生成潜在变量"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """
        前向传播，提取特征（均值 mu 或潜在变量 z）
        :param x: 输入数据 [batch_size, input_dim=9]
        :param return_features: 是否返回最后一层分类器之前的特征向量
        :return: logits（未经 Sigmoid）、mu、log_var
        """
        # 确保输入数据在正确的设备上
        device = x.device

        # 1. 线性投影到 d_model 维度
        x_proj = self.feature_proj(x)  # [batch_size, d_model]

        # 2. Transformer 处理
        x_trans = x_proj.unsqueeze(1)  # 变成 [batch_size, seq_len=1, d_model]
        x_trans = self.transformer_encoder(x_trans)  # 仍然是 [batch_size, seq_len=1, d_model]
        x_trans = x_trans.squeeze(1)  # 变回 [batch_size, d_model]

        # 3. 拼接 Transformer 输出和原始输入
        x_concat = torch.cat([x_trans, x], dim=-1)  # [batch_size, d_model + input_dim]

        # 4. 计算 VAE 均值和方差
        mu = self.fc_mu(x_concat)  # [batch_size, latent_dim]
        log_var = self.fc_logvar(x_concat)  # [batch_size, latent_dim]

        # 5. 采样潜在变量 z
        z = self.reparameterize(mu, log_var)  # [batch_size, latent_dim]

        # 6. 交叉注意力模块
        # 将 Transformer 输出和 z 调整到相同的维度
        z = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        x_trans = x_trans.unsqueeze(1)  # [batch_size, 1, d_model]

        # 添加线性层将 x_trans 转换为 latent_dim 维度
        x_trans = x_trans.view(x_trans.size(0), -1)  # 展平
        x_trans = self.x_trans_linear(x_trans)
        x_trans = x_trans.unsqueeze(1)  # 恢复形状 [batch_size, 1, latent_dim]

        attn_output, _ = self.cross_attention(query=z, key=x_trans, value=x_trans)
        attn_output = attn_output.squeeze(1)  # [batch_size, latent_dim]

        # 7. 分类器
        logits_before_final= self.classifier(attn_output)  # [batch_size, 32]

        # 8. 残差连接 + 归一化
        z_residual = self.residual_connection(mu)  # 使用 mu 作为残差连接
        logits_before_final = self.norm(logits_before_final + z_residual)

        # 9. 最终输出
        logits = self.final_layer(logits_before_final)  # [batch_size, 1]

        return logits.squeeze(-1), mu, log_var

    def vae_loss(self, logits, y_true, mu, log_var):
        """
        综合损失函数，包括二元交叉熵损失和 KL 散度。
        """
        classification_loss = F.binary_cross_entropy_with_logits(logits, y_true, reduction='mean')

        # 计算 KL 散度
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

        beta = 0.5
        total_loss = classification_loss + beta * kl_divergence

        return total_loss, classification_loss, kl_divergence