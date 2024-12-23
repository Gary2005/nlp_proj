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

dailymail_truncate = load_from_disk("/root/proj/cnn_dailymail_sentences")
train_origin, test_origin = dailymail_truncate.train_test_split(test_size=0.1).values() 
train_dataset = ParaphraseDataset(train_origin["article"], train_origin["article"], tokenizer, max_len=50)
test_dataset = ParaphraseDataset(test_origin["article"], test_origin["article"], tokenizer, max_len=50)


train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


wandb.init(project="encoder_decoder", entity="tqx_wandb", name="embeddingmodel")


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

class BinaryLayer(nn.Module):
    def __init__(self, mid_embedding_size, binary_size):
        super(BinaryLayer, self).__init__()
        self.ff1 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.ff2 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.eb = nn.Linear(mid_embedding_size, binary_size)
        self.be = nn.Linear(binary_size, mid_embedding_size)
        self.ff3 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.ff4 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.round = PassthroughRound.apply


    def forward(self, x):
        x = F.relu(self.ff1(x) + x)
        x = F.relu(self.ff2(x) + x)
        binary = self.eb(x)
        binary = self.round(F.sigmoid(binary))
        x = F.relu(self.be(binary))
        x = F.relu(self.ff3(x) + x)
        x = self.ff4(x)
        return x

def PassthroughArgMax(Embedding, A):

    max_indices = torch.argmax(A, dim=1)
    MaxA = torch.zeros(A.shape).to('cuda')
    MaxA.scatter_(1, max_indices.unsqueeze(1), 1)
    
    A = A.unsqueeze(1)
    MaxA = MaxA.unsqueeze(1)

    CorrectEmbedding = torch.matmul(A, Embedding).squeeze(1)
    MaxEmbedding = torch.matmul(MaxA, Embedding).squeeze(1)

    return MaxEmbedding.detach() + (CorrectEmbedding - CorrectEmbedding.detach())

class Embedding_Embedding_Model(nn.Module):
    def __init__(self, length):
        super(Embedding_Embedding_Model, self).__init__()
        self.Encoder = CoderLayer(2048)
        self.Binary = [BinaryLayer(2048, len).to('cuda') for len in length]
        self.ToA = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, len(length)),
        )
        self.n = len(length)
        self.Decoder = CoderLayer(2048)

    def forward(self, enc_src):
        enc_src = self.Encoder(enc_src)
        
        A_logits = self.ToA(enc_src)
        A = F.softmax(A_logits, dim=1)
        
        
        result = [binary(enc_src) for binary in self.Binary]
        

        result = torch.stack(result, dim=1)

        enc_src = PassthroughArgMax(result, A)

        # print(enc_src.shape)

        enc_src = self.Decoder(enc_src)

        return enc_src, A_logits, A.argmax(dim=1), A
    
alpha = 0
beta = 0

length = [16, 32, 64, 128, 256, 512, 1024, 2048]
length_tensor = torch.tensor(length).to(device)

eemodel = Embedding_Embedding_Model(length).to(device)

length_penalty = torch.arange(8).to(device).float()

optimizer = optim.Adam(eemodel.parameters(), lr=1e-4)
mse_loss = nn.MSELoss()
criterion_mask = nn.CrossEntropyLoss()

for epoch in range(10):
    cnt = 0

    bar = tqdm(train_loader)

    for batch in bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        input_ids = input_ids.permute(1, 0)
        labels = labels.permute(1, 0)

        with torch.no_grad():
            enc_src = model.get_enc_src(input_ids).detach()

        optimizer.zero_grad()
        output, A_logits, A_target, A = eemodel(enc_src)

        loss1 = mse_loss(output, enc_src)

        loss2 = torch.matmul(A, length_penalty).mean() * alpha

        loss3 = criterion_mask(A_logits, A_target) * beta

        loss = loss1 +loss2 + loss3
        loss.backward()
        optimizer.step()

        length_cho = length_tensor[A_target]

        wandb.log({"loss1": loss1.item(), "loss1_log": math.log(loss1.item()), "average_length": length_cho.float().mean()})
        # wandb.log({"loss1_log": math.log(loss1.item()), "loss2_log": math.log(loss2.item()), "loss3_log": math.log(loss3.item()), "loss_log": math.log(loss.item()), "average_length": mask_breakpoint.float().mean()})
        bar.set_description(f"loss1: {loss1.item()}, loss1_log: {math.log(loss1.item())}, average_length: {length_cho.float().mean()}")
        # bar.set_description(f"loss1_log: {math.log(loss1.item())}, loss2_log: {math.log(loss2.item())}, loss3_log: {math.log(loss3.item())}, loss_log: {math.log(loss.item())}, average_length: {mask_breakpoint.float().mean()}")

        cnt += 1
        if cnt % 2048 == 1:

            print("enc_src", enc_src[0, :])
            print("output", output[0, :])

            sample_intput = input_ids[:, 0].view(-1, 1)
            print("input_ids", sample_intput[:, 0])
            input = tokenizer.decode(sample_intput[:, 0], skip_special_tokens=True)
            print("input:", input)
            sample_output = model.generate_from_enc_src(output[0, :].unsqueeze(0), 50, tokenizer.cls_token_id, tokenizer.sep_token_id)
            print("output_ids", sample_output[:, 0])
            output = tokenizer.decode(sample_output[:, 0], skip_special_tokens=True)
            print("output:", output)

            sample_output = model.generate_from_enc_src(enc_src[0, :].unsqueeze(0), 50, tokenizer.cls_token_id, tokenizer.sep_token_id)
            print("correct_output_ids", sample_output[:, 0])
            output = tokenizer.decode(sample_output[:, 0], skip_special_tokens=True)
            print("correct_output:", output)
            
            torch.save(eemodel.state_dict(), "./embeddingmodel.pth")

