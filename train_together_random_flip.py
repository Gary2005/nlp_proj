import torch
import torch.nn as nn
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import math
import random

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
    
    def generate_from_enc_src(self, enc_src, max_len, start_symbol, end_symbol):
        self.eval()
        N = enc_src.shape[0]
        src_seq_length = 50 # max_len

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
    
    def get_enc_src(self, src):
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

        return enc_src

    def forward_from_enc_src(self, enc_src, trg):

        src_seq_length = 50
        trg_seq_length, N = trg.shape
        trg_mask = self.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        enc_src = self.ff2(enc_src)
        enc_src = enc_src.reshape(N, src_seq_length, -1).permute(1, 0, 2)

        trg_positions = (torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device))
        trg_padding_mask = self.make_src_mask(trg).permute(1, 0)
        embed_trg = self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)
        out = self.decoder(embed_trg, enc_src, tgt_mask=trg_mask, tgt_key_padding_mask=trg_padding_mask)
        out = self.fc_out(out)

        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_model_path = './../google-bert/bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

model_path = "/root/proj/encoder_decoder/train2_no_binary.pth"
model = Transformer(src_vocab_size=tokenizer.vocab_size, trg_vocab_size=tokenizer.vocab_size, src_pad_idx=tokenizer.pad_token_id, trg_pad_idx=tokenizer.pad_token_id, device=device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)

max_len = 50

# input = "Selected runs are not logging media for the key input, but instead are logging values of type string."
# print("input: ", input)
# input_ids = tokenizer(input, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")["input_ids"]
# input_ids = input_ids.to(device).view(-1, 1)
# # print("input_ids:", input_ids)
# # print(input[:, 0])
# output_ids = model.generate_new(input_ids, max_len, tokenizer.cls_token_id, tokenizer.sep_token_id)
# # print("output_ids:", output_ids)
# output = tokenizer.decode(output_ids.squeeze(), skip_special_tokens=True)
# print("output: ", output)


class ParaphraseDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_len=50):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        src = self.tokenizer(self.input_texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        tgt = self.tokenizer(self.target_texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        
        return {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels": tgt["input_ids"].squeeze(0),
        }

dailymail_truncate = load_from_disk("/root/autodl-tmp/cnn_dailymail_sentences")
train_origin, test_origin = dailymail_truncate.train_test_split(test_size=0.1).values() 
train_dataset = ParaphraseDataset(train_origin["article"], train_origin["article"], tokenizer, max_len=50)
test_dataset = ParaphraseDataset(test_origin["article"], test_origin["article"], tokenizer, max_len=50)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)



class CoderLayer(nn.Module):
    def __init__(self, dim):
        super(CoderLayer, self).__init__()
        self.ff1 = nn.Linear(dim, dim)
        self.ff2 = nn.Linear(dim, dim)
        self.ff3 = nn.Linear(dim, dim)

    def forward(self, x):
        x = F.relu(self.ff1(x) + x)
        x = F.relu(self.ff2(x) + x)
        x = self.ff3(x)
        return x

def constrained_normalization(x):
    return F.softmax(x, dim=1)

class PassPollute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, binary, random_rate):
        device = binary.device
        noise = torch.bernoulli(torch.full(binary.shape, random_rate, device=device).float())
        binary = binary.long()
        noise = noise.long()
        binary = binary ^ noise
        return binary

    @staticmethod
    def backward(ctx, grad_output):
        # 直接返回梯度，因为操作不可导
        return grad_output, None

class BinaryLayer(nn.Module):
    def __init__(self, mid_embedding_size, binary_size, random_rate):
        super(BinaryLayer, self).__init__()
        self.ff1 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.ff2 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.eb = nn.Linear(mid_embedding_size, binary_size)
        self.be = nn.Linear(binary_size, mid_embedding_size)
        self.ff3 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.ff4 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.round = PassthroughRound.apply
        self.pollute = PassPollute.apply
        self.random_rate = random_rate


    def forward(self, x):
        x = F.relu(self.ff1(x) + x)
        x = F.relu(self.ff2(x) + x)
        binary = self.eb(x)
        binary = self.round(F.sigmoid(binary))
        binary = self.pollute(binary, self.random_rate).float()
        x = F.relu(self.be(binary))
        x = F.relu(self.ff3(x) + x)
        x = self.ff4(x)
        return x
    
class Embedding_Embedding_Model(nn.Module):
    def __init__(self, length, random_rate):
        super(Embedding_Embedding_Model, self).__init__()
        self.Encoder = CoderLayer(2048)
        self.Binary = BinaryLayer(2048, length, random_rate).to("cuda")
        self.Decoder = CoderLayer(2048)

    def forward(self, enc_src):
        enc_src = self.Encoder(enc_src)
        
        enc_src = self.Binary(enc_src)


        enc_src = self.Decoder(enc_src)

        return enc_src
    
class Combined_Model(nn.Module):
    def __init__(self, transformer_model, embedding_model):
        super(Combined_Model, self).__init__()
        self.transformer_model = transformer_model
        self.embedding_model = embedding_model
        self.device = "cuda"

    def forward(self, input_ids, labels):
        enc_src = self.transformer_model.get_enc_src(input_ids)
        enc_src = self.embedding_model(enc_src)
        output = self.transformer_model.forward_from_enc_src(enc_src, labels)
        return output
    
    def generate_new(self, src, max_len, start_symbol, end_symbol):
        self.eval() # src: (seq_len, N)
        N = src.shape[1]

        enc_src = self.transformer_model.get_enc_src(src)
        enc_src = self.embedding_model(enc_src)

        outputs = (torch.ones(1, N).to(self.device) * start_symbol).type(torch.long)
        for _ in range(max_len):
            
            out = self.transformer_model.forward_from_enc_src(enc_src, outputs)

            prob = out[-1, :, :].argmax(dim=1, keepdim=True)

            if prob == end_symbol:
                break

            outputs = torch.cat((outputs, prob), dim=0)

        return outputs
    
        
    
import argparse

parser = argparse.ArgumentParser(description="接收命令行参数")

# 定义参数 -a
parser.add_argument('-len', type=int, help="一个整数参数", required=True)
parser.add_argument('-random_rate', type=float, help="一个浮点数参数", required=True)

# 解析命令行参数
args = parser.parse_args()


eemodel = Embedding_Embedding_Model(args.len, args.random_rate).to(device)
eemodel.load_state_dict(torch.load(f"/root/autodl-tmp/embeddingmodel_flip_{args.len}_random_rate_{args.random_rate}.pth", map_location=device))

combined_model = Combined_Model(model, eemodel)
combined_model = combined_model.to(device)
combined_model.train()


criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(combined_model.parameters(), lr=1e-6)


wandb.init(project="encoder_decoder", entity="tqx_wandb", name=f"combined_model_flip_{args.len}_random_rate_{args.random_rate}")

for epoch in range(10):
    cnt = 0

    bar = tqdm(train_loader)

    for batch in bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        input_ids = input_ids.permute(1, 0)
        labels = labels.permute(1, 0)

        output = combined_model(input_ids, labels[:-1, :])

        output = output.view(-1, output.shape[2])
        labels = labels[1:, :].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item()})
        bar.set_description(f"loss: {loss.item()}")

        cnt += 1
        if cnt % 1024 == 1:
            sample_intput = input_ids[:, 0].view(-1, 1)
            print("input_ids", sample_intput[:, 0])
            input = tokenizer.decode(sample_intput[:, 0], skip_special_tokens=True)
            print("input:", input)
            sample_output = combined_model.generate_new(sample_intput, 50, tokenizer.cls_token_id, tokenizer.sep_token_id)
            print("output_ids", sample_output[:, 0])
            output = tokenizer.decode(sample_output[:, 0], skip_special_tokens=True)
            print("output:", output)
            combined_model.train()
            torch.save(combined_model.state_dict(), f"/root/autodl-tmp/combined_model_flip_{args.len}_random_rate_{args.random_rate}.pth")