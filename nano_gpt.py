import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, num_embeddings, head_size, block_size, dropout_prob=0.2):
        super().__init__()

        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        batch, time_, channels = x.shape
        key = self.key(x)
        query = self.query(x)
        attention_filter = query @ key.transpose(-2, -1) * (key.shape[-1] ** 0.5)
        attention_filter = attention_filter.masked_fill(
            self.tril[:time_, :time_] == 0, float("-inf")
        )
        attention_filter = F.softmax(attention_filter, dim=-1)
        attention_filter = self.dropout(attention_filter)

        value = self.value(x)
        return attention_filter @ value


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads, num_embeddings, head_size, block_size, dropout_prob=0.2
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(num_embeddings, head_size, block_size, dropout_prob)
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(num_heads * head_size, num_embeddings)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.projection(x))
        return out


class FeedForward(nn.Module):
    def __init__(self, num_embeddings, decompression_coeff=4, dropout_prob=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_embeddings, num_embeddings * decompression_coeff),
            nn.ReLU(),
            nn.Linear(num_embeddings * decompression_coeff, num_embeddings),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.mlp(x)


class AttentionBlock(nn.Module):
    def __init__(self, num_heads, num_embeddings, block_size, dropout_prob=0.2):
        super().__init__()
        head_size = num_embeddings // num_heads
        self.attention = MultiHeadAttention(
            num_heads, num_embeddings, head_size, block_size, dropout_prob
        )
        self.feed_forward = FeedForward(num_embeddings, 4, dropout_prob)
        self.norm_1 = nn.LayerNorm(num_embeddings)
        self.norm_2 = nn.LayerNorm(num_embeddings)

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))
        # x = self.norm_1(x + self.attention(x))
        x = x + self.feed_forward(self.norm_2(x))
        # x = self.norm_2(x + self.feed_forward(x))
        return x


class NanoGPT(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        num_heads,
        num_blocks,
        num_embeddings,
        block_size,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.block_size = block_size

        self.input_embedding = nn.Embedding(vocabulary_size, num_embeddings)
        self.position_embedding = nn.Embedding(block_size, num_embeddings)
        self.attention_blocks = nn.Sequential(
            *(
                AttentionBlock(num_heads, num_embeddings, block_size)
                for _ in range(num_blocks)
            )
        )
        self.norm = nn.LayerNorm(num_embeddings)
        self.linear = nn.Linear(num_embeddings, vocabulary_size)

    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        x = self.input_embedding(inputs)
        x += self.position_embedding(torch.arange(T, device=self.device))
        x = self.norm(self.attention_blocks(x))
        logits = self.linear(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs, max_new_tokens):
        result = inputs.clone()
        for _ in range(max_new_tokens):
            conditional_input = result[:, -self.block_size :]
            logits, _ = self(conditional_input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            result = torch.cat((result, new_token), dim=1)

        return result
