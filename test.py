import torch
import torch.nn as nn
from transformers import AutoTokenizer


class PassthroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rounded = torch.round(x.detach())
        return rounded + (x - x.detach())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class PassthroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rounded = torch.round(x)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=4, forward_expansion=4, heads=8, dropout=0.1, max_length=50, device="cuda", mid_embedding_size=2048):
        super(Transformer, self).__init__()
        self.device = device

        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.src_position_embedding = nn.Embedding(max_length, embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.trg_position_embedding = nn.Embedding(max_length, embed_size)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=heads,
                dim_feedforward=forward_expansion * embed_size,
                dropout=dropout
            ),
            num_layers=num_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_size,
                nhead=heads,
                dim_feedforward=forward_expansion * embed_size,
                dropout=dropout
            ),
            num_layers=num_layers
        )

        self.ff1 = nn.Sequential(
            nn.Linear(max_length * embed_size, mid_embedding_size),
            nn.ReLU(),
        )

        self.roundpass = PassthroughRound.apply

        self.ff2 = nn.Sequential(
            nn.Linear(mid_embedding_size, mid_embedding_size),
            nn.ReLU(),
            nn.Linear(mid_embedding_size, max_length * embed_size),
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        # 将输入中的 padding token 设置为 True 以便后面忽略
        src_mask = (src == self.src_pad_idx)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape
        src_positions = (torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device))
        embed_src = self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        src_padding_mask = self.make_src_mask(src).permute(1, 0)
        trg_mask = self.generate_square_subsequent_mask(trg_seq_length).to(self.device)
        enc_src = self.encoder(embed_src, src_key_padding_mask=src_padding_mask)

        enc_src = enc_src.permute(1, 0, 2).reshape(N, -1)
        enc_src = self.ff1(enc_src)
        # enc_src = self.roundpass(F.sigmoid(enc_src))
        enc_src = self.ff2(enc_src)
        enc_src = enc_src.reshape(N, src_seq_length, -1).permute(1, 0, 2)

        trg_positions = (torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device))
        trg_padding_mask = self.make_src_mask(trg).permute(1, 0)
        embed_trg = self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)
        out = self.decoder(embed_trg, enc_src, tgt_mask=trg_mask, tgt_key_padding_mask=trg_padding_mask)
        out = self.fc_out(out)

        return out

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_new(self, src, max_len, start_symbol, end_symbol):
        self.eval() # src: (seq_len, N)
        src_seq_length, N = src.shape
        # print("src_seq_length:", src_seq_length)
        # print("N:", N)
        src_positions = (torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device))
        embed_src = self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        src_padding_mask = self.make_src_mask(src).permute(1, 0)
        enc_src = self.encoder(embed_src, src_key_padding_mask=src_padding_mask)

        enc_src = enc_src.permute(1, 0, 2).reshape(N, -1)
        enc_src = self.ff1(enc_src)
        # enc_src = self.roundpass(F.sigmoid(enc_src))
        enc_src = self.ff2(enc_src)
        enc_src = enc_src.reshape(N, src_seq_length, -1).permute(1, 0, 2)

        outputs = (torch.ones(1, N).to(self.device) * start_symbol).type(torch.long)
        for _ in range(max_len):
            trg_positions = (torch.arange(0, outputs.shape[0]).unsqueeze(1).expand(outputs.shape[0], N).to(self.device))
            embed_trg = self.trg_word_embedding(outputs) + self.trg_position_embedding(trg_positions)

            trg_mask = self.generate_square_subsequent_mask(outputs.shape[0]).to(self.device)
            tgt_mask = self.make_src_mask(outputs).permute(1, 0)
            out = self.decoder(embed_trg, enc_src, tgt_mask=trg_mask, tgt_key_padding_mask=tgt_mask)
            out = self.fc_out(out)

            prob = out[-1, :, :].argmax(dim=1, keepdim=True)
            if prob == end_symbol:
                break

            outputs = torch.cat((outputs, prob), dim=0)

        return outputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_model_path = './../google-bert/bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

model_path = "/root/proj/encoder_decoder/train2_no_binary.pth"
model = Transformer(src_vocab_size=tokenizer.vocab_size, trg_vocab_size=tokenizer.vocab_size, src_pad_idx=tokenizer.pad_token_id, trg_pad_idx=tokenizer.pad_token_id, device=device)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

max_len = 50

input = "Selected runs are not logging media for the key input, but instead are logging values of type string."
print("input: ", input)
input_ids = tokenizer(input, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")["input_ids"]
input_ids = input_ids.to(device).view(-1, 1)
# print("input_ids:", input_ids)
# print(input[:, 0])
output_ids = model.generate_new(input_ids, max_len, tokenizer.cls_token_id, tokenizer.sep_token_id)
# print("output_ids:", output_ids)
output = tokenizer.decode(output_ids.squeeze(), skip_special_tokens=True)
print("output: ", output)
