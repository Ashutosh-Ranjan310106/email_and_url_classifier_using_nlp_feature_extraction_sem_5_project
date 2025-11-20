import os
import json
import matplotlib.pyplot as plt

def plot_run(run_folder):
    """
    Loads logs.json from the run folder
    and plots all types of graphs with try-except protection.
    """
    json_path = os.path.join(run_folder, "logs.json")
    
    if not os.path.exists(json_path):
        print("❌ logs.json not found in:", run_folder)
        return
    
    # Load logs
    with open(json_path, "r") as f:
        logs = json.load(f)

    batch_train_losses = logs.get("batch_train_losses", [])
    batch_train_accs = logs.get("batch_train_accs", [])
    epoch_train_losses = logs.get("epoch_train_losses", [])
    epoch_train_accs = logs.get("epoch_train_accs", [])
    epoch_val_losses = logs.get("epoch_val_losses", [])
    epoch_val_accs = logs.get("epoch_val_accs", [])
    batch_times = logs.get("batch_times", [])
    epoch_times = logs.get("epoch_times", [])

    # Make /graphs folder inside run folder
    graphs_folder = os.path.join(run_folder, "graphs")
    os.makedirs(graphs_folder, exist_ok=True)

    # ---- Safe Save Function ----
    def safe_plot(x, y, title, xlabel, ylabel, filename):
        try:
            if len(x) == 0 or len(y) == 0:
                print(f"⚠ Skipped {filename} — missing data")
                return
            plt.figure()
            plt.plot(x, y)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_folder, filename))
            plt.close()
            print(f"   ✔ Saved {filename}")
        except Exception as e:
            print(f"❌ Error saving {filename}:", e)

    print("\n📊 Generating graphs...\n")

    # ============================
    # 1. BATCH-WISE GRAPHS
    # ============================

    safe_plot(
        list(range(len(batch_train_losses))),
        batch_train_losses,
        "Batch-wise Training Loss",
        "Batch Index",
        "Loss",
        "batch_loss_vs_batch.png"
    )

    safe_plot(
        list(range(len(batch_train_accs))),
        batch_train_accs,
        "Batch-wise Training Accuracy",
        "Batch Index",
        "Accuracy",
        "batch_acc_vs_batch.png"
    )

    # Cumulative time
    batch_cum_time = []
    total = 0
    for t in batch_times:
        total += t
        batch_cum_time.append(total)

    safe_plot(
        batch_cum_time,
        batch_train_losses,
        "Batch-wise Loss vs Time",
        "Time (seconds)",
        "Loss",
        "batch_loss_vs_time.png"
    )

    safe_plot(
        batch_cum_time,
        batch_train_accs,
        "Batch-wise Accuracy vs Time",
        "Time (seconds)",
        "Accuracy",
        "batch_acc_vs_time.png"
    )

    # ============================
    # 2. EPOCH-WISE GRAPHS
    # ============================

    safe_plot(
        list(range(len(epoch_train_losses))),
        epoch_train_losses,
        "Epoch-wise Train Loss",
        "Epoch",
        "Loss",
        "epoch_train_loss_vs_epoch.png"
    )

    safe_plot(
        list(range(len(epoch_train_accs))),
        epoch_train_accs,
        "Epoch-wise Train Accuracy",
        "Epoch",
        "Accuracy",
        "epoch_train_acc_vs_epoch.png"
    )

    safe_plot(
        list(range(len(epoch_val_losses))),
        epoch_val_losses,
        "Epoch-wise Validation Loss",
        "Epoch",
        "Loss",
        "epoch_val_loss_vs_epoch.png"
    )

    safe_plot(
        list(range(len(epoch_val_accs))),
        epoch_val_accs,
        "Epoch-wise Validation Accuracy",
        "Epoch",
        "Accuracy",
        "epoch_val_acc_vs_epoch.png"
    )

    # ---- Epoch time cumulative ----
    epoch_cum_time = []
    total = 0
    for t in epoch_times:
        total += t
        epoch_cum_time.append(total)

    safe_plot(
        epoch_cum_time,
        epoch_train_losses,
        "Train Loss vs Time (Epoch-wise)",
        "Time (seconds)",
        "Loss",
        "epoch_train_loss_vs_time.png"
    )

    safe_plot(
        epoch_cum_time,
        epoch_train_accs,
        "Train Accuracy vs Time (Epoch-wise)",
        "Time (seconds)",
        "Accuracy",
        "epoch_train_acc_vs_time.png"
    )

    safe_plot(
        epoch_cum_time,
        epoch_val_losses,
        "Val Loss vs Time (Epoch-wise)",
        "Time (seconds)",
        "Loss",
        "epoch_val_loss_vs_time.png"
    )

    safe_plot(
        epoch_cum_time,
        epoch_val_accs,
        "Val Accuracy vs Time (Epoch-wise)",
        "Time (seconds)",
        "Accuracy",
        "epoch_val_acc_vs_time.png"
    )

    print("\n✅ All available graphs saved in:", graphs_folder)
