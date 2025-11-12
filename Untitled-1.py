# %%
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from urllib.parse import urlparse
import random
tqdm.pandas()
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"✅ Seed fixed to {seed}")

# %%
# Load
encoded_data = torch.load("Dataset\encoded_data.pt",  weights_only=False)
print("✅ Encoded data loaded successfully")

# %%
from torch.utils.data import TensorDataset, DataLoader
# ============================================================
# Convert to TensorDataset and DataLoader
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

dataloader_dict = {}

def make_tensor_dataset(df):
    url_tensor = torch.stack(list(df["encode"]))
    labels_tensor = torch.tensor(df["label"].values, dtype=torch.long)
    return TensorDataset(url_tensor, labels_tensor)

for name, splits in encoded_data.items():
    dataloader_dict[name] = {}
    print(f"\n📦 Creating DataLoaders for {name}...")
    
    train_set = make_tensor_dataset(splits["train"])
    val_set = make_tensor_dataset(splits["valid"])
    test_set = make_tensor_dataset(splits["test"])
    
    dataloader_dict[name]["train_loader"] = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_dict[name]["val_loader"] = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dataloader_dict[name]["test_loader"] = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"✅ DataLoaders ready for {name} (Train/Val/Test)")

print("\n🚀 All DataLoaders are ready in `dataloader_dict`!")

# Example Access:
# dataloader_dict["dataset1"]["train_loader"]


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# 🔹 TinyByT5 Encoder (Reduced depth for short sequences)
# =====================================================
class TinyByT5Encoder(nn.Module):
    def __init__(self,
                 vocab_size=256,
                 d_model=128,
                 num_layers=2,
                 num_heads=2,
                 dim_ff=256,
                 max_len=100,
                 n_out=128,
                 dropout=0.1):
        super().__init__()

        # 🔹 Byte embedding layer (0–255)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 🔹 Positional embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # 🔹 Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 🔹 Final normalization
        self.final_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(in_features=d_model, out_features=n_out)

    def forward(self, x):
        """
        x: (batch, seq_len) — byte indices [0–255]
        """
        batch_size, seq_len = x.size()
        device = x.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(x) + self.pos_embedding(positions)

        #x = self.encoder(x)
        #x = self.embedding(x)
        x = self.projection(self.final_norm(x))

        return x  # (B, L, d_model)

# %%
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        w = x.mean(dim=2)                          # Global Average Pooling -> (B, C)
        w = F.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        w = w.unsqueeze(2)                         # (B, C, 1)
        return x * w                               # scale features

# %%
# 🔹 Residual Depthwise-Separable Multi-Kernel Block
class ResidualConvBlockDW(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sizes=[3, 5, 7], reduction=16):
        super().__init__()
        self.branches = nn.ModuleList()

        mid_ch = max(in_ch // 16, 8)  # reduce dimension before heavy convs

        for k in kernel_sizes:
            branch = nn.Sequential(
                # (B) Reduce channels first
                nn.Conv1d(in_ch, mid_ch, kernel_size=1, bias=False),
                #nn.BatchNorm1d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Dropout1d(0.25),

                # (A) Depthwise conv
                nn.Conv1d(mid_ch, mid_ch, kernel_size=k, padding=k // 2, groups=mid_ch, bias=False),
                #nn.BatchNorm1d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Dropout1d(0.25),

                # Pointwise to expand to out_ch
                nn.Conv1d(mid_ch, out_ch, kernel_size=1, bias=False),
                #nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)

        # Combine all kernel branches
        self.merge_conv = nn.Conv1d(out_ch * len(kernel_sizes), out_ch, kernel_size=1, bias=False)
        #self.merge_bn = nn.BatchNorm1d(out_ch)

        self.se = SEBlock(out_ch, reduction)
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # Parallel multi-kernel branches
        out = [branch(x) for branch in self.branches]
        out = torch.cat(out, dim=1)

        out = self.merge_conv(out)
        #out = self.merge_bn(out)
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)

    


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class URLBinaryCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, maxlen=100):
        super(URLBinaryCNN, self).__init__()
        self.maxlen = maxlen

        self.transformer_layer = nn.ModuleDict({
            "transformer": TinyByT5Encoder(vocab_size=vocab_size, max_len=maxlen, d_model=512, n_out=embed_dim)
        })

        # 2️⃣ Shared Layer (Global)
        self.shared_layer = nn.ModuleDict({
            "conv": ResidualConvBlockDW(embed_dim, 128),
            "bilstm": nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=True),
            "layer_norm": nn.LayerNorm(128 * 2),
            "fc1": nn.Linear(128 * 2, 128),
            "relu1": nn.ReLU(),
            "dropout1": nn.Dropout(0.5),
            "fc2": nn.Linear(128, 64),
            "relu2": nn.ReLU(),
            "dropout2": nn.Dropout(0.5),
        })

        # 3️⃣ Personalization Layer (Local)
        self.personal_layer = nn.ModuleDict({
            "head": nn.Linear(64, 1)
        })

    def forward(self, x):
        # Transformer
        x = self.transformer_layer["transformer"](x)
        x = x.permute(0, 2, 1)
        # Shared layers
        x = self.shared_layer["conv"](x)
        x = x.permute(0, 2, 1)
        x, _ = self.shared_layer["bilstm"](x)
        x = self.shared_layer["layer_norm"](x)
        x = x[:, -1, :]
        x = self.shared_layer["dropout2"](self.shared_layer["relu2"](self.shared_layer["fc2"](
            self.shared_layer["dropout1"](self.shared_layer["relu1"](self.shared_layer["fc1"](x)))
        )))
        # Personalization head
        x = self.personal_layer["head"](x)
        return torch.sigmoid(x)
    
    def extract_features(self, x):
        """Return deep features before final FC layers."""
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.conv_layer1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]          # shape: [batch, 32]
        x = self.flatten(x) 
        x = self.dropout(self.relu(self.fc1(x)))
        return x



# %%
import torch
import torch.nn as nn
import torch.optim as optim
import sys

class Train:
    def __init__(self, 
                 model, 
                 criterion, 
                 transformer_optimizer=None,
                 main_optimizer = None, 
                 scheduler_t=None,
                 scheduler_c=None,
                 train_loader=None, 
                 val_loader=None,   
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        """
        optimizer_groups: dict with keys like {"transformer": optimizer1, "cnn": optimizer2}
        schedulers: dict with keys matching optimizer_groups (optional)
        """
        self.model = model
        self.criterion = criterion

        self.transformer_params = []
        self.cnn_params = []
        for name, param in model.transformer_layer.named_parameters():
            self.transformer_params.append(param)
        for name, param in model.shared_layer.named_parameters():
            self.cnn_params.append(param)
        for name, param in model.personal_layer.named_parameters():
            self.cnn_params.append(param)



        self.transformer_optimizer = optim.NAdam(self.transformer_params, lr=1e-4) if transformer_optimizer is None  else transformer_optimizer
        self.main_optimizer = optim.NAdam(self.cnn_params, lr=1e-3) if main_optimizer is None else main_optimizer
        self.scheduler_t = optim.lr_scheduler.ReduceLROnPlateau(self.transformer_optimizer, mode='min', factor=0.5, patience=2) if scheduler_t is None  else scheduler_t
        self.scheduler_c = optim.lr_scheduler.ReduceLROnPlateau(self.main_optimizer, mode='min', factor=0.5, patience=2) if scheduler_c is None  else scheduler_c
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []


    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True


    def train(self, epochs_list=[3,3,4], early_stopping=True, frac=1.0, val_frac=1.0, alt_cycle = 2,start=0,  log=0):
        for phase, epochs in enumerate(epochs_list):
            for epoch in range(epochs):
                phase=3
                if phase == 0:
                    # 🧠 Train Transformer — freeze CNN/LSTM layers
                    for name, module in self.model.named_children():
                        if "transformer" not in name:
                            self.freeze_module(module)
                        else:
                            self.unfreeze_module(module)
                    active_optims = [self.transformer_optimizer]
                    active_scheds = [self.scheduler_t]
                    phase_name = "Embeding"
                elif phase == 1:
                    # 🎯 Train CNN/LSTM/FC — freeze Transformer

                    for name, module in self.model.named_children():
                        if "transformer" in name:
                            self.freeze_module(module)
                        else:
                            self.unfreeze_module(module)
                    active_optims = [self.main_optimizer]
                    active_scheds = [self.scheduler_c]
                    phase_name = "CNN"
                else:

                    for name, module in self.model.named_children():
                        self.unfreeze_module(module)
                    active_optims = [self.transformer_optimizer, self.main_optimizer]
                    active_scheds = [self.scheduler_t, self.scheduler_c]
                    phase_name = "CNN + Embeding"



                self.model.train()
                train_loss, correct_train, total_train = 0, 0, 0
                max_batches = int(len(self.train_loader) * (frac+start)) +1 
                start_batch = int(len(self.train_loader) * start)
                
                for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                    if batch_idx < start_batch:
                        continue
                    elif batch_idx >= max_batches:
                        break
                    
                    
                    batch_x, batch_y = batch_x.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True).float().unsqueeze(1)
                    
                    for opt in active_optims:
                        opt.zero_grad()

                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()

                    for opt in active_optims:
                        opt.step()




                    # === Metrics ===
                    batch_loss = loss.item()
                    preds = (outputs >= 0.5).float()
                    batch_acc = (preds == batch_y).float().mean().item()

                    if log >= 1 and (batch_idx + 1) % (20/log) == 0:
                        print(f"\rEpoch {epoch+1}/{epochs}: Training {phase_name} | "
                            f"Batch {batch_idx+1}/{max_batches} | "
                            f"Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}", end='')


                if log >= 2:
                    print(f'\r total training batch size {max_batches-start_batch}'.ljust(100), end='')
                    with torch.no_grad():
                        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                            if batch_idx < start_batch:
                                continue
                            elif batch_idx >= max_batches:
                                break
                            batch_x, batch_y = batch_x.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True).float().unsqueeze(1)

                            outputs = self.model(batch_x)
                            loss = self.criterion(outputs, batch_y)

                            batch_loss = loss.item()
                            preds = (outputs >= 0.5).float()
                            batch_acc = (preds == batch_y).float().mean().item()

                            train_loss += batch_loss * batch_x.size(0)
                            correct_train += (preds == batch_y).sum().item()
                            total_train += batch_x.size(0)

                        avg_train_loss = train_loss / total_train
                        train_acc = correct_train / total_train
                        self.train_losses.append(avg_train_loss)
                        self.train_accs.append(train_acc)


                    # === Validation ===
                    if self.val_loader is not None:
                        avg_val_loss, val_acc = self.evaluate(val_frac)
                        self.val_losses.append(avg_val_loss)
                        self.val_accs.append(val_acc)

                        print(f"\rEpoch {epoch+1}/{epochs} Training {phase_name}| "
                            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    else:
                        print(f"\rEpoch {epoch+1}/{epochs} Training {phase_name}| "
                            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")


                for sched in active_scheds:
                    sched.step(avg_val_loss)



    def evaluate(self, frac=1.0, start=0, log=2):
        self.model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        max_batches = max(int(len(self.val_loader) * (frac+start)), 0)+1
        start_batch = int(len(self.val_loader) * start)
        if log > 2:
            print('\rstarting from batch', start_batch, 'ending to batch', max_batches, f'total validation batch size {max_batches}')
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(self.val_loader):
                if batch_idx < start_batch:
                    continue
                elif batch_idx >= max_batches:
                    break
                batch_x, batch_y = batch_x.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True).float().unsqueeze(1)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                avg_batch_loss = loss.item()
                val_loss += avg_batch_loss * batch_x.size(0)
                preds = (outputs >= 0.5).float()
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)
        avg_val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        return avg_val_loss, val_acc
    



# %%
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from lion_pytorch import Lion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 🔧 Training Config
# ============================================================
num_epochs = [10,0,0]
lr = 0.001

# Store all dataset metrics
all_results = {}
vocab_size=97
# ============================================================
# 🔁 Training Loop for Each Dataset
# ============================================================
nn_model = {}
for dataset_name, loaders in dataloader_dict.items():
    print("\n" + "="*70)
    print(f"🚀 Training model on {dataset_name.upper()} dataset")
    print("="*70)
    for frac in  [0.1]:
        print(f"🧩 Using {frac*100:.0f}% of training data".center(50, '_'))
        train_loader = loaders["train_loader"]
        val_loader = loaders["val_loader"]

        # Initialize model, loss, optimizer
        nn_model[dataset_name] = URLBinaryCNN(vocab_size=vocab_size).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.NAdam(nn_model[dataset_name].parameters(), lr=lr, weight_decay=lr/10)
        trainer = Train(nn_model[dataset_name], criterion, train_loader=train_loader, val_loader=val_loader, main_optimizer=optimizer)
        # Lists to track performance
        trainer.train(num_epochs,frac=frac,val_frac=frac, log=2)

    

    # Store all results for this dataset
    all_results[dataset_name] = {
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses,
        "train_accs": trainer.train_accs,
        "val_accs": trainer.val_accs,
        "final_val_acc": trainer.val_accs[-1],
        "final_val_loss": trainer.val_losses[-1]
    }

print("\n✅ All datasets trained successfully!")

# ============================================================
# 📊 Summary of All Results
# ============================================================
print("\n" + "="*70)
print("📈 Final Validation Accuracy Summary")
print("="*70)
for name, res in all_results.items():
    print(f"{name:<20} | Val Acc: {res['final_val_acc']:.4f} | Val Loss: {res['final_val_loss']:.4f}")


# %%
torch.save(model.state_dict(), "url_cnn_lstm.pth")


# %%
#xgbost model + cnn based dl model
def extract_features(self, x):
        """Return deep features before final FC layers."""
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x3 = F.relu(self.conv1_3x3(x))
        x5 = F.relu(self.conv1_5x5(x))
        x7 = F.relu(self.conv1_7(x))
        x = torch.cat([x3, x5, x7], dim=1)
        x = F.relu(self.conv1_1x1(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]          # shape: [batch, 32]
        return x

# %%
import numpy as np
from tqdm import tqdm
import torch

model.eval()
features, labels = [], []

# 🔹 Extract CNN features for training set
with torch.no_grad():
    for x_batch, y_batch in tqdm(train_loader, desc="🔍 Extracting Train Features", unit="batch"):
        x_batch = x_batch.to(device, non_blocking=True)
        feats = nn_model[dataset_name].extract_features(x_batch)
        features.append(feats.cpu().numpy())
        labels.append(y_batch.cpu().numpy())

X_train = np.concatenate(features, axis=0)
y_train = np.concatenate(labels, axis=0)

# Free memory before val extraction
del features, labels, x_batch, y_batch, feats
torch.cuda.empty_cache()

# 🔹 Extract CNN features for validation set
features, labels = [], []
with torch.no_grad():
    for x_batch, y_batch in tqdm(val_loader, desc="🔍 Extracting Val Features", unit="batch"):
        x_batch = x_batch.to(device, non_blocking=True)
        feats = nn_model[dataset_name].extract_features(x_batch)
        features.append(feats.cpu().numpy())
        labels.append(y_batch.cpu().numpy())

X_val = np.concatenate(features, axis=0)
y_val = np.concatenate(labels, axis=0)


# %%
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    eval_metric="logloss",
    max_depth = 10,
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)


# %%
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

y_pred_prob = xgb_model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("F1:", f1_score(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_pred_prob))


# %%
#random forest
from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(
    n_estimators=100,   
    max_depth=30,     
    random_state=42
)



rfc_model.fit(X_train, y_train)

# %%
y_pred_prob = rfc_model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("F1:", f1_score(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_pred_prob))

# %%
#adding xgboost with cnn with handcrafted features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import xgboost as xgb
import pandas as pd




# %%
import pandas as pd
import re
import math
from urllib.parse import urlparse
import tldextract
from rapidfuzz import process, fuzz



import pickle
f= open('tld_encoding_serise_dataset_1.bin','rb')
tld_stats = pickle.load(file=f)
f.close()
print(type(tld_stats))

import kagglehub
import os
# Download latest version
folder_path = kagglehub.dataset_download("cheedcheed/top1m")

print("Path to dataset files:", folder_path)
csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

if not csv_files:
    raise FileNotFoundError("No CSV files found in the folder!")

# read the first CSV file
file_path = os.path.join(folder_path, csv_files[0])
alexa_top_1m_domain = pd.read_csv(file_path,header=None,names=['rank', 'domain'])
alexa_domains_set = set(alexa_top_1m_domain['domain'].apply(str.lower))

# --- Helper function: Shannon entropy ---
def safe_parse(url: str):
    """Safely parse URLs, adding http:// if missing and handling bad IPv6 parts."""
    if not isinstance(url, str) or not url.strip():
        return urlparse("http://")  

    # Ensure scheme exists
    if not re.match(r'^[a-zA-Z]+://', url):
        url = 'http://' + url

    # Clean invalid brackets that trigger IPv6 errors
    url = re.sub(r'\[.*?\]', '', url)

    try:
        return urlparse(url)
    except ValueError:
        # fallback: strip more aggressively if still malformed
        url = re.sub(r'[^a-zA-Z0-9:/._\-?&=]', '', url)
        return urlparse(url)
def calculate_entropy(string):
    """Measures randomness of characters in the URL."""
    if not string:
        return 0
    freq = {char: string.count(char) for char in set(string)}
    entropy = -sum((count / len(string)) * math.log2(count / len(string)) for count in freq.values())
    return entropy

# --- Main feature extraction function ---
def extract_handcrafted_features(url):
    features = {}
    if not re.match(r'^[hH]+[tT]+[tT]+[pP]+[sS]+://', url):
        url = 'http://' + url
    parsed = safe_parse(url)
    
    # 1️⃣ Basic structural features
    features['url_length'] = len(url)
    features['hostname_length'] = len(parsed.netloc)
    features['path_length'] = len(parsed.path)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_letters'] = sum(c.isalpha() for c in url)
    features['num_params'] = url.count('?')
    features['num_equals'] = url.count('=')
    features['num_slashes'] = url.count('/')
    features['num_at'] = url.count('@')

    # 2️⃣ Lexical / composition cues
    features['has_https'] = 1 if url.lower().startswith('https') else 0
    features['has_ip'] = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', parsed.netloc) else 0
    features['has_subdomain'] = 1 if parsed.netloc.count('.') > 1 else 0
    features['has_suspicious_words'] = 1 if re.search(r'(login|secure|verify|update|free|bank|click)', url.lower()) else 0

    # 3️⃣ Domain / TLD features
    extracted = tldextract.extract(url)
    main_domain = f"{extracted.domain}.{extracted.suffix}"
    if ':' in main_domain:  # remove port
        main_domain = main_domain.split(':')[0]
    features['domain_length'] = len(main_domain)
    features['in_alexa_top1m'] = 1 if main_domain in alexa_domains_set else 0
    '''
    if features['in_alexa_top1m'] == 0 and main_domain:  # only check if domain not in top1M
        # find closest match in Alexa domains
        best_match, score, _ = process.extractOne(main_domain, alexa_domains_set, scorer=fuzz.ratio)
        features['closest_alexa_domain'] = best_match
        features['closest_alexa_score'] = score  # 0-100
    else:
        features['closest_alexa_score'] = 1000  # high score to show that it is original url
    '''
    ext = tldextract.extract(url)
    tld = ext.suffix    # "com", "co.uk", "org"
    features['tld'] = tld if tld else 'unknown'
    features['tld_phish_ratio'] = tld_stats['phish_ratio'].get(features['tld'], 0.5)
    features['tld_total_frequency'] = tld_stats['total'].get(features['tld'], 1)

    # 4️⃣ Ratios
    features['digit_ratio'] = features['num_digits'] / (features['url_length'] + 1e-5)
    features['special_char_ratio'] = (features['num_hyphens'] + features['num_dots'] + features['num_slashes']) / (features['url_length'] + 1e-5)

    # 5️⃣ Entropy (measures randomness / obfuscation)
    features['url_entropy'] = calculate_entropy(url)

    # 6️⃣ Misplacement indicators
    # '@' symbol used to hide real domain (like "http://evil.com@legit.com")
    features['at_in_domain'] = 1 if '@' in parsed.netloc else 0
    
    # Double slashes '//' appearing after path (used to trick users)
    features['double_slash_in_path'] = 1 if re.search(r'/.+//', parsed.path) else 0

    return features

# %%
from tqdm import tqdm
tqdm.pandas()
for i in encoded_data:

    for split in encoded_data[i]:
        print("feature extracting", i, split)
        encoded_data[i][split] = encoded_data[i][split].assign(**encoded_data[i][split].url.progress_apply(lambda url : pd.Series(extract_handcrafted_features(url))))

# %%
# Split features and labels
independent_features = ['url_length', 'hostname_length', 'path_length',
    'num_dots', 'num_hyphens', 'num_digits', 'num_letters', 'num_params',
    'num_equals', 'num_slashes', 'num_at', 'has_https', 'has_ip',
    'has_subdomain', 'has_suspicious_words', 'domain_length',
    'in_alexa_top1m', 'tld_phish_ratio', 'tld_total_frequency',
    'digit_ratio', 'special_char_ratio', 'url_entropy', 'at_in_domain',
    'double_slash_in_path']
dependet_features  = 'label'
model_dict = {}
for i in encoded_data:
    print(f"train for {i}")
    X_train = encoded_data[i]['train'][independent_features]
    y_train = encoded_data[i]['train'][dependet_features]
    X_test = encoded_data[i]['valid'][independent_features]
    y_test = encoded_data[i]['valid'][dependet_features]

    # Create XGBoost classifier
    model_dict[i] = xgb.XGBClassifier(
        n_estimators=100,      # number of boosting rounds
        learning_rate=0.01,     # step size shrinkage
        max_depth=30,           # tree depth
        eval_metric='logloss', # evaluation metric
        use_label_encoder=False
    )

    # Train the model
    model_dict[i].fit(X_train, y_train)
    # Predict
    y_pred = model_dict[i] .predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


# %%
#logistic_regression(nn+xgboost)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
independent_features = ['url_length', 'hostname_length', 'path_length',
    'num_dots', 'num_hyphens', 'num_digits', 'num_letters', 'num_params',
    'num_equals', 'num_slashes', 'num_at', 'has_https', 'has_ip',
    'has_subdomain', 'has_suspicious_words', 'domain_length',
    'in_alexa_top1m', 'tld_phish_ratio', 'tld_total_frequency',
    'digit_ratio', 'special_char_ratio', 'url_entropy', 'at_in_domain',
    'double_slash_in_path']
dependent_features  = 'label'
logistic_model = {}
for name in encoded_data:
    #print(model_dict[name])
    #print(nn_model[name])
    nn_model[name].eval()
    with torch.no_grad():
        model = nn_model[name]
        #print(model)
        x_ = encoded_data[name]['train']['encode']
        x_np = np.stack(x_.values)
        x_ = torch.tensor(x_np, dtype=torch.long).to(device)  # move once to GPU/CPU where model is
        batch_size = 512  # adjust for your GPU memory
        nn_preds_train = []

        # 🔹 Process manually in batches
        for i in tqdm(range(0, len(x_), batch_size), desc=f"Predicting NN for {name}", ncols=80):
            batch_x = x_[i:i + batch_size]
            outputs = model(batch_x)
            nn_preds_train.append(outputs)

    # 🔹 Combine all predictions
    nn_preds_train = torch.cat(nn_preds_train, dim=0).cpu()

    # XGBoost predictions
    xgb_preds_train = model_dict[name].predict_proba(encoded_data[name]['train'][independent_features])[:, 1]

    # Stack predictions as new features
    meta_X = np.column_stack((nn_preds_train, xgb_preds_train))
    meta_y = encoded_data[name]['train'][dependent_features]


    meta_model= LogisticRegressionCV(
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    penalty="l2",
    scoring="roc_auc",
    solver="lbfgs",
    Cs=10,
    max_iter=1000,
    n_jobs=-1
)
    meta_model.fit(meta_X, meta_y)


    with torch.no_grad():
        x_ = encoded_data[name]['valid']['encode']
        x_np = np.stack(x_.values)
        x_ = torch.tensor(x_np, dtype=torch.long).to(device)  # move once to GPU/CPU where model is
        batch_size = 512  # adjust for your GPU memory
        nn_preds_val = []

        # 🔹 Process manually in batches
        for i in tqdm(range(0, len(x_), batch_size), desc=f"Predicting NN for {name}", ncols=80):
            batch_x = x_[i:i + batch_size]
            outputs = model(batch_x)
            nn_preds_val.append(outputs)

    # 🔹 Combine all predictions
    nn_preds_val = torch.cat(nn_preds_val, dim=0).cpu()
    # XGBoost predictions
    xgb_preds_val = model_dict[name].predict_proba(encoded_data[name]['valid'][independent_features])[:, 1]

    # Stack predictions as new features
    meta_X_val = np.column_stack((nn_preds_val, xgb_preds_val))
    meta_y_val = encoded_data[name]['valid'][dependent_features]


    y_pred_train = meta_model.predict(meta_X)
    y_pred_prob_train = meta_model.predict_proba(meta_X)[:, 1]

    print("\n📘 TRAIN METRICS")
    print("Accuracy :", accuracy_score(meta_y, y_pred_train))
    print("Precision:", precision_score(meta_y, y_pred_train))
    print("Recall   :", recall_score(meta_y, y_pred_train))
    print("F1-score :", f1_score(meta_y, y_pred_train))
    print("ROC AUC  :", roc_auc_score(meta_y, y_pred_prob_train))

    # ===============================
    # 🔹 Validation Metrics
    # ===============================
    y_pred_val = meta_model.predict(meta_X_val)
    y_pred_prob_val = meta_model.predict_proba(meta_X_val)[:, 1]

    print("\n📗 VALIDATION METRICS")
    print("Accuracy :", accuracy_score(meta_y_val, y_pred_val))
    print("Precision:", precision_score(meta_y_val, y_pred_val))
    print("Recall   :", recall_score(meta_y_val, y_pred_val))
    print("F1-score :", f1_score(meta_y_val, y_pred_val))
    print("ROC AUC  :", roc_auc_score(meta_y_val, y_pred_prob_val))



# %%
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
independent_features = ['url_length', 'hostname_length', 'path_length',
    'num_dots', 'num_hyphens', 'num_digits', 'num_letters', 'num_params',
    'num_equals', 'num_slashes', 'num_at', 'has_https', 'has_ip',
    'has_subdomain', 'has_suspicious_words', 'domain_length',
    'in_alexa_top1m', 'tld_phish_ratio', 'tld_total_frequency',
    'digit_ratio', 'special_char_ratio', 'url_entropy', 'at_in_domain',
    'double_slash_in_path']
dependent_features  = 'label'
for name in encoded_data:
    #print(model_dict[name])
    #print(nn_model[name])
    nn_model[name].eval()
    with torch.no_grad():
        model = nn_model[name]
        #print(model)
        x_ = encoded_data[name]['train']['encode']
        x_np = np.stack(x_.values)
        x_ = torch.tensor(x_np, dtype=torch.long).to(device)  # move once to GPU/CPU where model is
        batch_size = 512  # adjust for your GPU memory
        nn_preds_train = []

        # 🔹 Process manually in batches
        for i in tqdm(range(0, len(x_), batch_size), desc=f"Predicting NN for {name}", ncols=80):
            batch_x = x_[i:i + batch_size]
            outputs = model(batch_x)
            nn_preds_train.append(outputs)

    # 🔹 Combine all predictions
    nn_preds_train = torch.cat(nn_preds_train, dim=0).cpu()

    # XGBoost predictions
    xgb_preds_train = model_dict[name].predict_proba(encoded_data[name]['train'][independent_features])[:, 1]

    # Stack predictions as new features
    meta_X = np.column_stack((nn_preds_train, xgb_preds_train))
    meta_y = encoded_data[name]['train'][dependent_features]


    params = {
    "hidden_layer_sizes": [(8,), (16,), (24,)],
    "alpha": [1e-3, 1e-4, 1e-5],  # L2 regularization
    "learning_rate_init": [1e-3],
    "early_stopping": [True],
    "max_iter": [500],
    }

    mlp = MLPClassifier(random_state=42)
    meta_model = GridSearchCV(mlp, params, cv=StratifiedKFold(3), scoring="accuracy", n_jobs=-1, verbose=3)
    meta_model.fit(meta_X, meta_y)


    with torch.no_grad():
        x_ = encoded_data[name]['valid']['encode']
        x_np = np.stack(x_.values)
        x_ = torch.tensor(x_np, dtype=torch.long).to(device)  # move once to GPU/CPU where model is
        batch_size = 512  # adjust for your GPU memory
        nn_preds_val = []

        # 🔹 Process manually in batches
        for i in tqdm(range(0, len(x_), batch_size), desc=f"Predicting NN for {name}", ncols=80):
            batch_x = x_[i:i + batch_size]
            outputs = model(batch_x)
            nn_preds_val.append(outputs)

    # 🔹 Combine all predictions
    nn_preds_val = torch.cat(nn_preds_val, dim=0).cpu()
    # XGBoost predictions
    xgb_preds_val = model_dict[name].predict_proba(encoded_data[name]['valid'][independent_features])[:, 1]

    # Stack predictions as new features
    meta_X_val = np.column_stack((nn_preds_val, xgb_preds_val))
    meta_y_val = encoded_data[name]['valid'][dependent_features]


    y_pred_train = meta_model.predict(meta_X)
    y_pred_prob_train = meta_model.predict_proba(meta_X)[:, 1]

    print("\n📘 TRAIN METRICS")
    print("Accuracy :", accuracy_score(meta_y, y_pred_train))
    print("Precision:", precision_score(meta_y, y_pred_train))
    print("Recall   :", recall_score(meta_y, y_pred_train))
    print("F1-score :", f1_score(meta_y, y_pred_train))
    print("ROC AUC  :", roc_auc_score(meta_y, y_pred_prob_train))

    # ===============================
    # 🔹 Validation Metrics
    # ===============================
    y_pred_val = meta_model.predict(meta_X_val)
    y_pred_prob_val = meta_model.predict_proba(meta_X_val)[:, 1]

    print("\n📗 VALIDATION METRICS")
    print("Accuracy :", accuracy_score(meta_y_val, y_pred_val))
    print("Precision:", precision_score(meta_y_val, y_pred_val))
    print("Recall   :", recall_score(meta_y_val, y_pred_val))
    print("F1-score :", f1_score(meta_y_val, y_pred_val))
    print("ROC AUC  :", roc_auc_score(meta_y_val, y_pred_prob_val))



# %%
#xgboost(nn+xgboost)
from sklearn.linear_model import LogisticRegression

independent_features = ['url_length', 'hostname_length', 'path_length',
    'num_dots', 'num_hyphens', 'num_digits', 'num_letters', 'num_params',
    'num_equals', 'num_slashes', 'num_at', 'has_https', 'has_ip',
    'has_subdomain', 'has_suspicious_words', 'domain_length',
    'in_alexa_top1m', 'tld_phish_ratio', 'tld_total_frequency',
    'digit_ratio', 'special_char_ratio', 'url_entropy', 'at_in_domain',
    'double_slash_in_path']
dependent_features  = 'label'
logistic_model = {}
for name in encoded_data:
    #print(model_dict[name])
    #print(nn_model[name])
    nn_model[name].eval()
    with torch.no_grad():
        model = nn_model[name]
        #print(model)
        x_ = encoded_data[name]['train']['encode']
        x_np = np.stack(x_.values)
        x_ = torch.tensor(x_np, dtype=torch.long).to(device)  # move once to GPU/CPU where model is
        batch_size = 512  # adjust for your GPU memory
        nn_preds_train = []

        # 🔹 Process manually in batches
        for i in tqdm(range(0, len(x_), batch_size), desc=f"Predicting NN for {name}", ncols=80):
            batch_x = x_[i:i + batch_size]
            outputs = model(batch_x)
            nn_preds_train.append(outputs)

    # 🔹 Combine all predictions
    nn_preds_train = torch.cat(nn_preds_train, dim=0).cpu()

    # XGBoost predictions
    xgb_preds_train = model_dict[name].predict_proba(encoded_data[name]['train'][independent_features])[:, 1]

    # Stack predictions as new features
    meta_X = np.column_stack((nn_preds_train, xgb_preds_train))
    meta_y = encoded_data[name]['train'][dependent_features]

    with torch.no_grad():
        x_ = encoded_data[name]['valid']['encode']
        x_np = np.stack(x_.values)
        x_ = torch.tensor(x_np, dtype=torch.long).to(device)  # move once to GPU/CPU where model is
        batch_size = 512  # adjust for your GPU memory
        nn_preds_val = []

        # 🔹 Process manually in batches
        for i in tqdm(range(0, len(x_), batch_size), desc=f"Predicting NN for {name}", ncols=80):
            batch_x = x_[i:i + batch_size]
            outputs = model(batch_x)
            nn_preds_val.append(outputs)

    # 🔹 Combine all predictions
    nn_preds_val = torch.cat(nn_preds_val, dim=0).cpu()
    # XGBoost predictions
    xgb_preds_val = model_dict[name].predict_proba(encoded_data[name]['valid'][independent_features])[:, 1]

    # Stack predictions as new features
    meta_X_val = np.column_stack((nn_preds_val, xgb_preds_val))
    meta_y_val = encoded_data[name]['valid'][dependent_features]


    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        bjective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        max_depth = 10,
        random_state=42,
        
    )

    xgb_model.fit(
        meta_X, meta_y,
        #eval_set=[(meta_X_val, meta_y_val)],
    )


    y_pred_train = meta_model.predict(meta_X)
    y_pred_prob_train = meta_model.predict_proba(meta_X)[:, 1]

    print("\n📘 TRAIN METRICS")
    print("Accuracy :", accuracy_score(meta_y, y_pred_train))
    print("Precision:", precision_score(meta_y, y_pred_train))
    print("Recall   :", recall_score(meta_y, y_pred_train))
    print("F1-score :", f1_score(meta_y, y_pred_train))
    print("ROC AUC  :", roc_auc_score(meta_y, y_pred_prob_train))

    # ===============================
    # 🔹 Validation Metrics
    # ===============================
    y_pred_val = meta_model.predict(meta_X_val)
    y_pred_prob_val = meta_model.predict_proba(meta_X_val)[:, 1]

    print("\n📗 VALIDATION METRICS")
    print("Accuracy :", accuracy_score(meta_y_val, y_pred_val))
    print("Precision:", precision_score(meta_y_val, y_pred_val))
    print("Recall   :", recall_score(meta_y_val, y_pred_val))
    print("F1-score :", f1_score(meta_y_val, y_pred_val))
    print("ROC AUC  :", roc_auc_score(meta_y_val, y_pred_prob_val))


# %% [markdown]
# # Fedrated learning

# %%
#train global model on little part(5%) of dataset
dataset_name = 'Dataset 3 (kmack/Phishing_urls)'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = URLBinaryCNN(vocab_size=len(vocab)).to(device)
loaders = dataloader_dict[dataset_name]
train_loader = loaders["train_loader"]
val_loader = loaders["val_loader"]
lr  = 0.01
print(f"🧩 Using {0.05*100:.0f}% of training data".center(50, '_'))

criterion = nn.BCELoss()
optimizer = torch.optim.NAdam(global_model.parameters(), lr=lr, weight_decay=lr/10)
trainer = Train(global_model, criterion, train_loader=train_loader, val_loader=val_loader)
# Lists to track performance
trainer.train(epochs_list=[3,3,4],frac=0.05,val_frac=0.05, log=2)



# %%
import numpy as np
# using alpha to produce unidentical splits high alpha more identical
num_clients = 5
alpha = 0.5
total_data = 0.3
client_fractions = np.random.dirichlet([alpha] * num_clients) * total_data
print(client_fractions)
print(client_fractions.sum())


# %%
client_models = []
for i in range(num_clients):
    client_model = URLBinaryCNN(vocab_size=len(vocab)).to(device)                     # fresh instance
    client_model.load_state_dict(global_model.state_dict())
    client_models.append(client_model)

# %%
print('clints initial accuracy')
start = 0
for i, model in enumerate(client_models):
    criterion = nn.BCELoss()
    trainer = Train(model, criterion=criterion, train_loader=train_loader, val_loader=val_loader)
    avg_val_loss, val_acc = trainer.evaluate(frac=client_fractions[i], start=start, log=2)
    start += client_fractions[i]
    print(avg_val_loss, val_acc)

# %%
# trainig clints
start = 0
for i, model in enumerate(client_models):
    criterion = nn.BCELoss()
    trainer = Train(model, criterion=criterion, train_loader=train_loader, val_loader=val_loader)
    trainer.train(epochs_list=[2,0,0],frac=client_fractions[i], val_frac=client_fractions[i], start=start, log=2)
    start += client_fractions[i]


# %%
# tansfering weights and averaging them
import copy
client_weights = [client_model.state_dict() for client_model in client_models]
total_samples = sum(client_fractions)

new_global_state = copy.deepcopy(global_model.state_dict())

# set all params to zero before summing
for key in new_global_state.keys():
    new_global_state[key] = torch.zeros_like(new_global_state[key])

# aggregate client updates
for client_model, n_i in zip(client_models, client_fractions):
    client_state = client_model.state_dict()
    for key in new_global_state.keys():
        new_global_state[key] += client_state[key] * (n_i / total_samples)

# load averaged weights back into global model
global_model.load_state_dict(new_global_state)



# %%
trainer = Train(global_model, criterion, train_loader=train_loader, val_loader=val_loader)
avg_val_loss, val_acc = trainer.evaluate(frac=0.3, start=0, log=2)
print(avg_val_loss, val_acc)

# %%
import copy
def make_clients(num_clients=5, alpha=0.5, total_data=1):
    import numpy as np
    # using alpha to produce unidentical splits high alpha more identical
    client_fractions = np.random.dirichlet([alpha] * num_clients) * total_data
    client_models = []
    for i in range(num_clients):
        client_model = URLBinaryCNN(vocab_size=len(vocab)).to(device)                     # fresh instance
        client_model.load_state_dict(global_model.state_dict())
        client_models.append(client_model)
    return client_model, client_fractions
def train_client(client_models, client_fractions, start=0):
    for i, _ in enumerate(client_models):
        criterion = nn.BCELoss()
        trainer = Train(client_models[i], criterion=criterion, train_loader=train_loader, val_loader=val_loader)
        trainer.train(epochs_list=[2,0,0],frac=client_fractions[i], val_frac=client_fractions[i], start=start, log=2)
        start += client_fractions[i]
def fed_avg(client_models, client_fractions):
    total_samples = sum(client_fractions)
    new_global_state = copy.deepcopy(global_model.state_dict())
    # set all params to zero before summing
    for key in new_global_state.keys():
        new_global_state[key] = torch.zeros_like(new_global_state[key])
    # aggregate client updates
    for client_model, n_i in zip(client_models, client_fractions):
        client_state = client_model.state_dict()
        for key in new_global_state.keys():
            new_global_state[key] += client_state[key] * (n_i / total_samples)
    # load averaged weights back into global model
    global_model.load_state_dict(new_global_state)
def update_clients(client_models):
    for client_model in client_models:
        client_model.load_state_dict(global_model.state_dict())
client_model, client_fractions = make_clients(5, total_data=0.3) 

# %%
for i in range(3):
    train_client(client_models, client_fractions, start=0.3)   
    fed_avg(client_models, client_fractions)
    update_clients(client_models)
    trainer = Train(global_model, criterion, train_loader=train_loader, val_loader=val_loader)
    avg_val_loss, val_acc = trainer.evaluate(frac=0.3, start=0, log=2)
    print(f'global model loss {avg_val_loss} accuracy {val_acc}')

# %% [markdown]
# ## fedrated learning with persnalization
# # meta learing
# 
# 

# %%
#train global model on little part(5%) of dataset
dataset_name = 'Dataset 3 (kmack/Phishing_urls)'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = URLBinaryCNN(vocab_size=len(vocab)).to(device)
loaders = dataloader_dict[dataset_name]
train_loader = loaders["train_loader"]
val_loader = loaders["val_loader"]
lr  = 0.01
print(f"🧩 Using {0.05*100:.0f}% of training data".center(50, '_'))

criterion = nn.BCELoss()
optimizer = torch.optim.NAdam(global_model.parameters(), lr=lr, weight_decay=lr/10)
trainer = Train(global_model, criterion, train_loader=train_loader, val_loader=val_loader)
# Lists to track performance
trainer.train(epochs_list=[3,3,4],frac=0.05,val_frac=0.05, log=2)



# %%
import copy
def make_clients(num_clients=5, alpha=0.5, total_data=1):
    import numpy as np
    # using alpha to produce unidentical splits high alpha more identical
    client_fractions = np.random.dirichlet([alpha] * num_clients) * total_data
    client_models = []
    for i in range(num_clients):
        client_model = URLBinaryCNN(vocab_size=len(vocab)).to(device)                     # fresh instance
        client_model.load_state_dict(global_model.state_dict())
        client_models.append(client_model)
    return client_models, client_fractions
def train_client(client_models, client_fractions, start=0):
    for i, _ in enumerate(client_models):
        print("training model: ",i)
        criterion = nn.BCELoss()
        trainer = Train(client_models[i], criterion=criterion, train_loader=train_loader, val_loader=val_loader)
        trainer.train(epochs_list=[2,0,0],frac=client_fractions[i], val_frac=client_fractions[i], start=start, log=2)
        start += client_fractions[i]
def fed_meta_persnalization_learing(global_model, client_models, client_fractions, meta_lr=0.1):
    total_samples = sum(client_fractions)
    weights = [n_i / total_samples for n_i in client_fractions]
    global_state = {
        **global_model.transformer_layer.state_dict(),
        **global_model.shared_layer.state_dict()
    }

    avg_state = copy.deepcopy(global_state)
    for key in avg_state.keys():
        avg_state[key] = torch.zeros_like(avg_state[key])
        for client_model, w in zip(client_models, weights):
            client_state = {
                **client_model.transformer_layer.state_dict(),
                **client_model.shared_layer.state_dict()
            }
            avg_state[key] += client_state[key] * w

    new_state = {}
    for key in global_state.keys():
        new_state[key] = global_state[key] + meta_lr * (avg_state[key] - global_state[key])

    # 5️⃣ Load updated parameters into global model
    global_model.transformer_layer.load_state_dict({
        k: v for k, v in new_state.items() if k in global_model.transformer_layer.state_dict()
    })
    global_model.shared_layer.load_state_dict({
        k: v for k, v in new_state.items() if k in global_model.shared_layer.state_dict()
    })
def update_clients(client_models, global_model):
    for client in client_models:
        client.transformer_layer.load_state_dict(global_model.transformer_layer.state_dict())
        client.shared_layer.load_state_dict(global_model.shared_layer.state_dict())


# %%
client_models, client_fractions = make_clients(10, total_data=0.3) 
print(client_fractions)
print('clints initial accuracy')
start = 0.3
for i, model in enumerate(client_models):
    criterion = nn.BCELoss()
    trainer = Train(model, criterion=criterion, train_loader=train_loader, val_loader=val_loader)
    avg_val_loss, val_acc = trainer.evaluate(frac=client_fractions[i], start=start, log=3)
    start += client_fractions[i]
    print(avg_val_loss, val_acc, client_fractions[i])

# %%
trainer = Train(global_model, criterion, train_loader=train_loader, val_loader=val_loader)
for i in range(3):
    train_client(client_models, client_fractions, start=0.3)   
    fed_meta_persnalization_learing(global_model, client_models, client_fractions)
    update_clients(client_models, global_model)
    avg_val_loss, val_acc = trainer.evaluate(frac=0.3, start=0, log=2)
    print(f'global model loss {avg_val_loss} accuracy {val_acc}')

# %%


# %%


# %%


# %%



