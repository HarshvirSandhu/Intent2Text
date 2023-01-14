import math
import spacy
import torch
import torch.nn as nn
import torchtext
import pandas as pd
# df=pd.read_csv('C:/Users/harsh/Downloads/spoc/train/spoc-train.tsv', sep='\t')
# print(df.head(), df[df.keys()[0]])
# print(df[df.keys()[0]][0], df[df.keys()[1]][0])
import json

# Pre-Processing
# path = 'C:/Users/harsh/PycharmProjects/Harshvir_S/Txt2Code/conala-corpus-v1.1/conala-corpus/conala-train.json'
# with open('new_train.json', 'w', encoding='utf=8') as f:
#     for sample in json.load(open(path, 'r')):
#         if sample['rewritten_intent'] is not None:
#             sample = str(json.dumps(sample)) + '\n'
#             f.write(sample)

# tok = lambda x: x.split()
def rm_braces(x):
    return str(x).strip('[]')
text_eng = torchtext.data.Field(sequential=True, use_vocab=True, tokenize='spacy',
                                tokenizer_language='en_core_web_sm', stop_words=['[', ']', '\''])
text_correct = torchtext.data.Field(sequential=True, use_vocab=True, tokenize='spacy', tokenizer_language='en_core_web_sm')

fields = [('concepts', text_eng), ('target', text_eng)]

data = torchtext.data.TabularDataset(path='data_nlp.csv',
                                     format='csv', fields=fields)
# text_correct.build_vocab(data)
text_eng.build_vocab(data)
data_loader = torchtext.data.BucketIterator(dataset=data, batch_size=32)
# print(text_eng.vocab.stoi)
for i in data_loader:
    print(i.concepts.shape)
    print('concepts')
    for word in i.concepts.T[0]:
        print(text_eng.vocab.itos[word], end='')
    print('\n target')
    for word in i.target.T[0]:
        print(text_eng.vocab.itos[word], end=' ')
    print('\n')
    break


loss_list = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_src_dim = 288
embed_trg_dim = 288
nhead = 12
num_encoder_layers = 6
num_decoder_layers = 6
dim_feed_forward = 4
dropout = 0.1
src_vocab_size = len(text_eng.vocab)
trg_vocab_size = len(text_eng.vocab)
src_pad_idx = text_eng.vocab.stoi['<pad>']


def translate_sentence(model, sentence, target, english, device, max_length=50):
    # Load german tokenizer
    spacy_eng = spacy.load("en_core_web_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_eng(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, english.init_token)
    tokens.append(english.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [english.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [target.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == target.vocab.stoi["<eos>"]:
            break
    # print(output)

    translated_sentence = [target.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


class MyModel(nn.Module):
    def __init__(self, inp_vocab_size, out_vocab_size, embed_dim, nheads, num_encoder_layers, num_decoder_layers,
                 dropout, src_pad_idx, dim_feedforward):
        super(MyModel, self).__init__()
        self.src_embed_layer = nn.Embedding(inp_vocab_size, embed_dim)
        self.trg_embed_layer = nn.Embedding(out_vocab_size, embed_dim)
        self.transformer = nn.Transformer(embed_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout)
        self.src_pad_idx = src_pad_idx
        self.drop = nn.Dropout(dropout)
        self.last = nn.Linear(embed_dim, out_vocab_size)
        self.embed_dim = embed_dim

    def positional_encoding(self, x):
        seq_len = x.shape[0]
        indices = torch.arange(seq_len).unsqueeze(1)
        pe = torch.zeros_like(x)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim))
        # print(pe.shape, indices.shape, div_term.shape)
        pe[:, 0, 0::2] = torch.sin(indices * div_term)
        pe[:, 0, 1::2] = torch.cos(indices * div_term)
        return pe.to(device)

    def make_src_mask(self, src):
        mask = src.transpose(0, 1) == self.src_pad_idx
        return mask

    def forward(self, src, trg):
        embed_src = self.src_embed_layer(src)
        embed_src += self.positional_encoding(embed_src)
        embed_src = self.drop(embed_src)

        embed_trg = self.drop(self.trg_embed_layer(trg))

        mask_src = self.make_src_mask(src).to(device)
        mask_trg = self.transformer.generate_square_subsequent_mask(trg.shape[0]).to(device)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=mask_src, tgt_mask=mask_trg)
        out = self.last(out)
        return out


model = MyModel(embed_dim=embed_src_dim, num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers, nheads=nhead, dim_feedforward=dim_feed_forward, dropout=dropout,
                inp_vocab_size=src_vocab_size, out_vocab_size=trg_vocab_size, src_pad_idx=src_pad_idx,
                ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=text_eng.vocab.stoi['<pad>'])
sentence = 'monkey, forest, banana'
num_epochs = 20
print(len(data_loader))
for epoch in range(1, num_epochs):
    epoch_loss = 0
    model.train()
    for num, i in enumerate(data_loader):
        i.concepts = i.concepts.to(device)
        i.target = i.target.to(device)
        output = model(i.concepts, i.target[:-1])
        if num >= 200 and num % 200 == 0:
            print(epoch, '---', num)
        output = output.reshape(-1, output.shape[2])
        tgt = i.target[1:].reshape(-1)
        # print(trg.shape, output.shape)
        loss = criterion(output, tgt)
        # print(loss)
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
    loss_list.append(epoch_loss)
    print(epoch, epoch_loss, min(loss_list))
    model.eval()
    print(translate_sentence(model=model, sentence=sentence, target=text_eng, english=text_eng, device=device))