import torch
import torch.nn as nn
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import wandb

class PassthroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rounded = torch.round(x.detach())
        return rounded + (x - x.detach())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class EncoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EncoderLayer, self).__init__()
        self.ff1 = nn.Linear(input_dim, output_dim)
        self.ff2 = nn.Linear(output_dim, output_dim)
        self.ff3 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.ff1(x))
        x = F.relu(self.ff2(x) + x)
        x = F.relu(self.ff3(x) + x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DecoderLayer, self).__init__()
        self.ff1 = nn.Linear(input_dim, input_dim)
        self.ff2 = nn.Linear(input_dim, input_dim)
        self.ff3 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.ff1(x) + x)
        x = F.relu(self.ff2(x) + x)
        x = self.ff3(x)
        return x

def constrained_normalization(x):
    positive_x = F.relu(x) + 1e-6
    return positive_x / positive_x.sum(dim=1, keepdim=True)

class BinaryLayer(nn.Module):
    def __init__(self, mid_embedding_size, hidden_size):
        super(BinaryLayer, self).__init__()

        self.to_mask1 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.to_mask2 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.to_mask3 = nn.Linear(mid_embedding_size, mid_embedding_size)

        self.to_binary1 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.to_binary2 = nn.Linear(mid_embedding_size, mid_embedding_size)
        self.to_binary3 = nn.Linear(mid_embedding_size, hidden_size)

    def forward(self, x):
        mask = F.relu(self.to_mask1(x) + x)
        mask = F.relu(self.to_mask2(mask) + mask)
        mask = self.to_mask3(mask)

        binary = F.relu(self.to_binary1(x) + x)
        binary = F.relu(self.to_binary2(binary) + binary)
        binary = self.to_binary3(binary)

        return mask, binary
        

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0.1, max_length=50, device="cuda", mid_embedding_size=1024, hidden_size=8192):
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

        self.ff1 = EncoderLayer(max_length * embed_size, mid_embedding_size)

        self.ff1 = nn.Sequential(
            nn.Linear(max_length * embed_size, mid_embedding_size),
            nn.ReLU(),
        )

        self.roundpass = PassthroughRound.apply
        self.binary = BinaryLayer(mid_embedding_size, hidden_size)

        self.debinary = nn.Sequential(
            nn.Linear(hidden_size, mid_embedding_size),
            nn.ReLU(),
            nn.Linear(mid_embedding_size, mid_embedding_size),
            nn.ReLU(),
            nn.Linear(mid_embedding_size, mid_embedding_size),
        )
        
        self.ff2 = DecoderLayer(mid_embedding_size, max_length * embed_size)

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
        trg_positions = (torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device))

        embed_src = self.src_word_embedding(src) + self.src_position_embedding(src_positions)

        embed_src = embed_src.permute(1, 0, 2).reshape(N, -1)
        embed_src = self.ff1(embed_src)

        '''
        Binary Layer
        '''

        # mask, binary = self.binary(embed_src)
        # mask_prob = constrained_normalization(mask)
        # mask_prob_pre_sum = torch.cumsum(mask_prob, dim=1)
        # binary = F.sigmoid(binary)
        # binary = self.roundpass(binary)
        mask_prob = None
        mask_breakpoint = None
        # mask_binary = 1 - self.roundpass(mask_prob_pre_sum)
        # mask_breakpoint = torch.argmax((mask_prob_pre_sum >= 0.5).float(), dim=1)
        # embed_src = (binary * 2 - 1) * mask_binary
        # embed_src = self.debinary(binary)

        '''
        Binary Layer
        '''

        embed_src = self.ff2(embed_src)
        embed_src = embed_src.reshape(N, src_seq_length, -1).permute(1, 0, 2)

        embed_trg = self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)

        src_padding_mask = self.make_src_mask(src).permute(1, 0)
        trg_padding_mask = self.make_src_mask(trg).permute(1, 0)
        trg_mask = self.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        enc_src = self.encoder(embed_src, src_key_padding_mask=src_padding_mask)
        out = self.decoder(embed_trg, enc_src, tgt_mask=trg_mask, tgt_key_padding_mask=trg_padding_mask, memory_key_padding_mask=src_padding_mask)
        out = self.fc_out(out)

        return out, mask_prob, mask_breakpoint

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_new(self, src, max_len, start_symbol, end_symbol):
        self.eval() # src: (seq_len, N)
        src_seq_length, N = src.shape
        src_positions = (torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device))
        embed_src = self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        embed_src = embed_src.permute(1, 0, 2).reshape(N, -1)
        embed_src = self.ff1(embed_src)
        '''
        Binary Layer
        '''

        # mask, binary = self.binary(embed_src)
        # mask_prob = constrained_normalization(mask)
        # mask_prob_pre_sum = torch.cumsum(mask_prob, dim=1)
        # binary = F.sigmoid(binary)
        # binary = self.roundpass(binary)
        # mask_binary = 1 - self.roundpass(mask_prob_pre_sum)
        # mask_breakpoint = torch.argmax((mask_prob_pre_sum >= 0.5).float(), dim=1)
        # embed_src = (binary * 2 - 1) * mask_binary
        # embed_src = self.debinary(binary)

        '''
        Binary Layer
        '''
        embed_src = self.ff2(embed_src)
        embed_src = embed_src.reshape(N, src_seq_length, -1).permute(1, 0, 2)

        src_padding_mask = self.make_src_mask(src).permute(1, 0)
        enc_src = self.encoder(embed_src, src_key_padding_mask=src_padding_mask)

        outputs = (torch.ones(1, N).to(self.device) * start_symbol).type(torch.long)
        for _ in range(max_len):
            trg_positions = (torch.arange(0, outputs.shape[0]).unsqueeze(1).expand(outputs.shape[0], N).to(self.device))
            embed_trg = self.trg_word_embedding(outputs) + self.trg_position_embedding(trg_positions)

            trg_mask = self.generate_square_subsequent_mask(outputs.shape[0]).to(self.device)
            tgt_mask = self.make_src_mask(outputs).permute(1, 0)
            out = self.decoder(embed_trg, enc_src, tgt_mask=trg_mask, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
            out = self.fc_out(out)

            prob = out[-1, :, :].argmax(dim=1, keepdim=True)
            if prob == end_symbol:
                break

            outputs = torch.cat((outputs, prob), dim=0)

        return outputs


# 确保将模型和数据放置在适当的设备上（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

local_model_path = './../google-bert/bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(local_model_path)


# 初始化模型，具体参数根据实际需要设置
# 例如: vocab sizes, pad indices 等

mid_embedding_size = 2048
hidden_size = 16384
model = Transformer(src_vocab_size=tokenizer.vocab_size, trg_vocab_size=tokenizer.vocab_size, src_pad_idx=tokenizer.pad_token_id, trg_pad_idx=tokenizer.pad_token_id, device=device, mid_embedding_size=mid_embedding_size, hidden_size=hidden_size)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"训练参数总数: {total_params}")


class ParaphraseDataset(torch.utils.data.Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_len=50):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        # self.init_dataset()

    def init_dataset(self):
        print("init dataset")
        pbdr = tqdm(range(len(self.input_texts)))
        for i in pbdr:
            src = self.tokenizer(self.input_texts[i], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
            tgt = self.tokenizer(self.target_texts[i], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
            self.data.append({
                "input_ids": src["input_ids"].squeeze(0),
                "attention_mask": src["attention_mask"].squeeze(0),
                "labels": tgt["input_ids"].squeeze(0),
            })

    def __len__(self):
        return len(self.input_texts)
        # return len(self.data)

    def __getitem__(self, idx):
        # return self.data[idx]
        src = self.tokenizer(self.input_texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        tgt = self.tokenizer(self.target_texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        
        return {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels": tgt["input_ids"].squeeze(0),
        }

# def save_dataset(dataset, path):
#     data_dict = {
#         "input_ids": [item["input_ids"] for item in dataset.data],
#         "attention_mask": [item["attention_mask"] for item in dataset.data],
#         "labels": [item["labels"] for item in dataset.data],
#     }
#     dataset_hf = Dataset.from_dict(data_dict)
#     dataset_hf.save_to_disk(path)

print("initialize")
dailymail_truncate = load_from_disk("/root/proj/cnn_dailymail_sentences")
print("initialize 2")
train_origin, test_origin = dailymail_truncate.train_test_split(test_size=0.1).values() 
print("initialize 3")
train_dataset = ParaphraseDataset(train_origin["article"], train_origin["article"], tokenizer, max_len=50)
test_dataset = ParaphraseDataset(test_origin["article"], test_origin["article"], tokenizer, max_len=50)

# save_dataset(train_dataset, "/root/proj/encoder_decoder/train_dataset")
# save_dataset(test_dataset, "/root/proj/encoder_decoder/test_dataset")

# train_dataset = load_from_disk("/root/proj/encoder_decoder/train_dataset")
# test_dataset = load_from_disk("/root/proj/encoder_decoder/test_dataset")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
# cosine distance 
criterion_cos = nn.CosineEmbeddingLoss()
criterion_mask = torch.nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)



wandb.init(project="encoder_decoder", entity="tqx_wandb", name="no_binary")


# hyper parameters

length_loss_weight = torch.arange(2048)
length_loss_weight = torch.exp((length_loss_weight / 2048) * 4) - 1
length_loss_weight = length_loss_weight.detach()
length_loss_weight = length_loss_weight.to(device)

beta = 1

# training loop
for epoch in range(10):
    bar = tqdm(train_loader)
    model.train()
    cnt = 0
    for batch in bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        input_ids = input_ids.permute(1, 0)
        labels = labels.permute(1, 0)

        output, mask_prob, mask_breakpoint = model(input_ids, labels[:-1, :])

        # print("mask", mask)
        # print("mask_prob", mask_prob)

        output = output.view(-1, output.shape[2])
        labels = labels[1:, :].reshape(-1)

        optimizer.zero_grad()
        loss1 = criterion(output, labels) 
        # loss2 = torch.matmul(mask_prob, length_loss_weight).mean() * 10
        # loss3 = criterion_mask(torch.log(mask_prob), mask_breakpoint) * 10
        # loss = loss1 + loss2 + loss3
        # loss = loss1 + loss2
        loss = loss1
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item()})
        wandb.log({"loss1": loss1.item()})
        # wandb.log({"loss2": loss2.item()})
        # wandb.log({"loss3": loss3.item()})
        # wandb.log({"average_length": mask_breakpoint.float().mean().item()})
        bar.set_description(f"loss: {loss.item()}, loss1: {loss1.item()}")

        cnt += 1
        if cnt % 2048 == 1:
            sample_intput = input_ids[:, 0].view(-1, 1)
            print("input_ids", sample_intput[:, 0])
            input = tokenizer.decode(sample_intput[:, 0], skip_special_tokens=True)
            print("input:", input)
            sample_output = model.generate_new(sample_intput, 50, tokenizer.cls_token_id, tokenizer.sep_token_id)
            print("output_ids", sample_output[:, 0])
            output = tokenizer.decode(sample_output[:, 0], skip_special_tokens=True)
            print("output:", output)
            model.train()
            torch.save(model.state_dict(), "./truncate_transformer_no_binary.pth")
            wandb.log({"input": input, "output": output})