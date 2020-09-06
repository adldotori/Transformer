import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.h = 8
        self.d_k = self.d_model // self.h
        self.d_v = self.d_model // self.h
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.h)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.h) 
        self.W_V = nn.Linear(self.d_model, self.d_v * self.h)
        self.W_O = nn.Linear(self.h * self.d_v, self.d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        matmul_qk = torch.matmul(Q, torch.transpose(K, -2, -1)) / self.d_k ** 0.5
        attention_weights = nn.Softmax(dim=-1)(matmul_qk)
        attention = torch.matmul(attention_weights, V)
        return attention, attention_weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view(batch_size, -1, self.h, self.d_k).transpose(-2, -3)
        K = self.W_K(K).view(batch_size, -1, self.h, self.d_k).transpose(-2, -3)
        V = self.W_V(V).view(batch_size, -1, self.h, self.d_v).transpose(-2, -3)

        attn, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        attn = attn.view(batch_size, -1, self.h * self.d_v)

        res = self.W_O(attn)
        return res

class FeedForwardNetworks(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 512
        self.d_ff = 2048

        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.multi_head_attention = MultiHeadAttention(self.d_model)
        self.feed_forward_networks = FeedForwardNetworks()
        self.layer_norm = nn.LayerNorm(self.d_model)
    
    def forward(self, x):
        x_attn = self.multi_head_attention(x, x, x)
        x = self.layer_norm(x + x_attn)
        x_ffn = self.feed_forward_networks(x)
        x = self.layer_norm(x + x_ffn)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.multi_head_attention = MultiHeadAttention(self.d_model)
        self.feed_forward_networks = FeedForwardNetworks()
        self.layer_norm = nn.LayerNorm(self.d_model)
    
    def forward(self, x, enc_in):
        x_attn = self.multi_head_attention(x, x, x)
        x = self.layer_norm(x + x_attn)
        x_attn = self.multi_head_attention(x, enc_in, enc_in)
        x = self.layer_norm(x + x_attn)
        x_ffn = self.feed_forward_networks(x)
        x = self.layer_norm(x + x_ffn)
        return x

def get_positional_encoding_table(length, d_model):
    encoding_dim = np.array([1/10000 ** (i/d_model) for i in range(d_model)])
    encoding_pos = np.array([encoding_dim * i for i in range(length)])   
    encoding_pos[:, ::2] = np.sin(encoding_pos[:, ::2])
    encoding_pos[:, 1::2] = np.cos(encoding_pos[:, 1::2])
    encoding_table = torch.Tensor(encoding_pos).to(device)
    return encoding_table

class Encoder(nn.Module):
    def __init__(self, vocab_size, n, p_drop=0.1):
        super().__init__()
        self.d_model = 512
        self.vocab_size = vocab_size
        self.n = n

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.encoder_layer = nn.ModuleList([EncoderLayer(self.d_model) for _ in range(self.n)])
     
    def forward(self, x):
        x = self.embedding(x)
        x = x + get_positional_encoding_table(x.shape[1], self.d_model)
        for layer in self.encoder_layer:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, n, p_drop=0.1):
        super().__init__()
        self.d_model = 512
        self.vocab_size = vocab_size
        self.n = n

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.encoder_layer = nn.ModuleList([DecoderLayer(self.d_model) for _ in range(self.n)])
        self.linear = nn.Linear(self.d_model, self.vocab_size)
     
    def forward(self, x, enc_in):
        x = self.embedding(x)
        x = x + get_positional_encoding_table(x.shape[1], self.d_model)
        for layer in self.encoder_layer:
            x = layer(x, enc_in)
        x = self.linear(x)
        return x

class Transformer(nn.Module):
    def __init__(self, kor_vocab_size, eng_vocab_size, n_enc=6, n_dec=6):
        super().__init__()

        self.encoder = Encoder(kor_vocab_size, n_enc)
        self.decoder = Decoder(eng_vocab_size, n_dec)

    def forward(self, inputs, outputs):
        inputs = torch.transpose(inputs, 0, 1)
        outputs = torch.transpose(outputs, 0, 1)
        enc = self.encoder(inputs)
        res = self.decoder(outputs, enc)
        return res

if __name__ == '__main__':
    batch_size = 1
    inputs = torch.randint(20, (3, batch_size)).to(device)
    outputs = torch.randint(18, (80, batch_size)).to(device)

    model = Transformer(kor_vocab_size=10000, eng_vocab_size=20000, n_enc=6, n_dec=6).to(device)
    res = model(inputs, outputs)
    print(res.shape)
