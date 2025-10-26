import kagglehub
from datasets import load_dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import csv

# -------------------- DOWNLOAD AND LOAD --------------------
dataset1_path = kagglehub.dataset_download("sid321axn/malicious-urls-dataset")
dataset2_path = kagglehub.dataset_download("ndarvind/phiusiil-phishing-url-dataset")

# Read CSVs
df1 = pd.read_csv(os.path.join(dataset1_path, [f for f in os.listdir(dataset1_path) if f.endswith('.csv')][0]))
df2 = pd.read_csv(os.path.join(dataset2_path, [f for f in os.listdir(dataset2_path) if f.endswith('.csv')][0]))

# Load HuggingFace phishing dataset
train_dataset = load_dataset("kmack/Phishing_urls", split="train")
test_dataset = load_dataset("kmack/Phishing_urls", split="test")
valid_dataset = load_dataset("kmack/Phishing_urls", split="valid")

df3_train = train_dataset.to_pandas()
df3_test = test_dataset.to_pandas()
df3_valid = valid_dataset.to_pandas()
df3 = pd.concat([df3_train, df3_test, df3_valid], ignore_index=True)



rows = []
with open(r'Dataset\grambeddings_dataset_main\train.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        # if more than 2 fields, merge all after the first one into URL
        if len(row) > 2:
            row = [row[0], ','.join(row[1:])]
        rows.append(row)

df4_train = pd.DataFrame(rows, columns=['label', 'url'])

rows = []
with open(r'Dataset\grambeddings_dataset_main\test.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        # if more than 2 fields, merge all after the first one into URL
        if len(row) > 2:
            row = [row[0], ','.join(row[1:])]
        rows.append(row)

df4_test = pd.DataFrame(rows, columns=['label', 'url'])

df4 = pd.concat([df4_train, df4_test], ignore_index=True)
print(df4.head())
# -------------------- STANDARDIZE COLUMN NAMES --------------------
def normalize_columns(df):
    df.columns = df.columns.str.lower()
    if "text" in df.columns:
        df.rename(columns={"text": "url"}, inplace=True)
    if "type" in df.columns:
        df.rename(columns={"type": "label"}, inplace=True)
    if "category" in df.columns:
        df.rename(columns={"category": "label"}, inplace=True)
    if "result" in df.columns:
        df.rename(columns={"result": "label"}, inplace=True)
    if "target" in df.columns:
        df.rename(columns={"target": "label"}, inplace=True)
    return df

df1 = normalize_columns(df1)
df2 = normalize_columns(df2)
df3 = normalize_columns(df3)
df4['label'][df4['label'] == 1] = 0 
df4['label'][df4['label'] == 2] = 1 
df4['label'] = df4['label'].astype(int)
# -------------------- FILTER ONLY BENIGN + PHISHING --------------------
def filter_and_encode(df, name):
    # Lowercase labels for consistency
    df['label'] = df['label'].astype(str).str.lower()

    # Keep only phishing and benign
    phishing_labels = ['phish', 'phishing', 'malicious', 'bad']  # allow variants
    benign_labels = ['benign', 'safe', 'legit', 'good']

    df = df[df['label'].isin(phishing_labels + benign_labels)]

    # Encode labels
    df['label'] = df['label'].apply(lambda x: 1 if x in phishing_labels else 0)

    print(f"\n✅ {name} cleaned: {len(df)} samples (Phishing={df['label'].sum()}, Benign={len(df)-df['label'].sum()})")

    return df

df1 = filter_and_encode(df1, "Dataset 1 (Malicious URLs)")

def drop_dublicates(df):
    df = df.dropna(subset=["url", "label"])
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    df = df[["url", "label"]]
    return df

df1 = drop_dublicates(df1)
df2 = drop_dublicates(df2)
df3 = drop_dublicates(df3)
df4 = drop_dublicates(df4)
# -------------------- SPLIT EACH DATASET --------------------
def split_dataset(df, name):
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    print(f"\n📂 {name} split → Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    return train_df, valid_df, test_df

# -------------------- SUMMARIZER FUNCTION --------------------
def summarize_dataset(name, train_df, valid_df, test_df):
    print(f"\n{'='*70}")
    print(f"📊 Summary for {name}")
    print(f"{'='*70}")
    print(train_df)
    
    def basic_summary(df, subset_name):
        print(type(df),df.info())
        print(f"\n🔹 {subset_name} Set:")
        print(f"Samples: {len(df):,}")
        print(f"Missing URLs: {df['url'].isna().sum()}")
        print(f"Duplicate URLs: {df['url'].duplicated().sum()}")
        print("Label distribution:")
        print(df['label'].value_counts(normalize=True).round(3).to_string())

    # Show summaries
    basic_summary(train_df, "Train")
    basic_summary(valid_df, "Validation")
    basic_summary(test_df, "Test")

    # Compare distributions
    print("\n🔍 Label Distribution Comparison (vs Train):")
    ref = train_df['label'].value_counts(normalize=True)
    for name_df, df in [("Validation", valid_df), ("Test", test_df)]:
        other = df['label'].value_counts(normalize=True)
        comp = pd.concat([ref, other], axis=1, keys=["Train", name_df]).fillna(0)
        comp["Difference"] = (comp[name_df] - comp["Train"]).round(3)
        print(f"\n{name_df} comparison:\n{comp}")


# -------------------- GENERATOR WRAPPERS --------------------
def lazy_dataframe(train1, valid1, test1, train2, valid2, test2, train3, valid3, test3, train4, valid4, test4):
    """Return a generator that yields the DataFrame only when iterated (lazy loading)."""
    def generator():
        yield train1, valid1, test1
        yield train2, valid2, test2
        yield train3, valid3, test3
        yield train4, valid4, test4
    return generator





all_dataset = lazy_dataframe(*split_dataset(df1, "Dataset 1 (Malicious URLs)"), *split_dataset(df2, "Dataset 1 (Malicious URLs)"), *split_dataset(df3, "Dataset 1 (Malicious URLs)"), *split_dataset(df4, "Dataset 1 (Malicious URLs)")  )


del df1, df2, df3, df4
del df3_test, df3_train, df3_valid
del df4_test, df4_train
del train_dataset, valid_dataset, test_dataset
# -------------------- SUMMARIZE EACH --------------------
if __name__ == "__main__":
    gen = all_dataset()
    print("✅ Paths Loaded Successfully:")
    print("Dataset 1 Path:", dataset1_path)
    print("Dataset 2 Path:", dataset2_path)

    summarize_dataset("Dataset 1 (Malicious URLs)", *next(gen))
    summarize_dataset("Dataset 2 (PHIUSIIL Phishing URLs)", *next(gen))
    summarize_dataset("Dataset 3 (Phishing URLs - HuggingFace)", *next(gen))
    summarize_dataset("Dataset 4 (grambeddings)", *next(gen))

