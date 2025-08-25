# tf_imdb_rnn.py
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Hyperparameters
vocab_size = 10000
max_len = 200
embedding_dim = 64
rnn_units = 64
batch_size = 64
epochs = 3

# 1. Load IMDb (keep top 10k words)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# 2. Pad/cut each review to length 200
x_train = pad_sequences(x_train, maxlen=max_len)
x_test  = pad_sequences(x_test,  maxlen=max_len)

print("Train shape:", x_train.shape, y_train.shape)
print("Test shape:", x_test.shape, y_test.shape)

# 3. Build RNN model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    SimpleRNN(rnn_units, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# 4. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Train
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    verbose=1
)

# 6. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"[TensorFlow] Test Accuracy: {acc:.4f}")

# Pytorch version
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Hyperparameters
vocab_size = 10000
embedding_dim = 64
hidden_size = 64
max_len = 200
batch_size = 64
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load raw text (label, text) using torchtext
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

# 2) Build vocabulary (limit to 10k tokens + <pad>)
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>"], max_tokens=vocab_size)
vocab.set_default_index(vocab["<pad>"])

# Recreate iterators because the first pass is exhausted
train_iter, test_iter = IMDB(split=('train', 'test'))

# 3) Encode function (text → tensor of token ids)
def encode(text):
    return torch.tensor(vocab(tokenizer(text)), dtype=torch.long)

# 4) Collate function for DataLoader (pad/cut to max_len)
def collate_batch(batch):
    label_map = {"neg": 0.0, "pos": 1.0}
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_map[label])
        token_ids = encode(text)[:max_len]             # cut if longer than max_len
        texts.append(token_ids)
    # pad to the length of the longest sequence in the batch
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    # if still shorter than max_len, pad more to reach 200
    if texts.size(1) < max_len:
        pad_len = max_len - texts.size(1)
        pad_tensor = torch.full((texts.size(0), pad_len), vocab["<pad>"], dtype=torch.long)
        texts = torch.cat([texts, pad_tensor], dim=1)
    elif texts.size(1) > max_len:
        texts = texts[:, :max_len]
    labels = torch.tensor(labels, dtype=torch.float32)
    return texts.to(device), labels.to(device)

# 5) DataLoaders
train_loader = DataLoader(list(IMDB(split='train')), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader  = DataLoader(list(IMDB(split='test')),  batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# 6) Define a simple RNN classifier
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)                # (B, T, E)
        output, hidden = self.rnn(embedded)         # output: (B, T, H), hidden: (1, B, H)
        last_hidden = hidden.squeeze(0)             # (B, H)
        logits = self.fc(last_hidden)               # (B, 1)
        probs = self.sigmoid(logits)                # (B, 1)
        return probs

model = RNNClassifier(vocab_size=len(vocab), embedding_dim=embedding_dim,
                      hidden_size=hidden_size, pad_idx=vocab["<pad>"]).to(device)

# 7) Loss & Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 8) Training loop (BPTT tự động bởi autograd)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts).squeeze(1)          # (B,)
        loss = criterion(outputs, labels)
        loss.backward()                             # BPTT
        optimizer.step()
        running_loss += loss.item()
    print(f"[PyTorch] Epoch {epoch+1}/{epochs} - Train Loss: {running_loss/len(train_loader):.4f}")

# 9) Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts).squeeze(1)          # (B,)
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"[PyTorch] Test Accuracy: {correct/total:.4f}")
