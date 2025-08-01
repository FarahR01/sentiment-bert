import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime
import os
import sys
import json
import platform

# Set multiprocessing start method for Windows safety
if platform.system() == "Windows":
    if hasattr(torch.multiprocessing, 'set_start_method'):
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.training.dataset import IMDBDataset
from app.training.model import load_model
from app.core.logger import configure_logging
# Experiment Tracking with Weights & Biases
import wandb
from torch.amp import GradScaler
from app.utils.log_confusion_matrix import log_confusion_matrix

# Determine device availability (safer check)
try:
    use_amp = torch.cuda.is_available()
    if use_amp:
        # Verify CUDA is actually usable
        torch.cuda.get_device_properties(0)
except (RuntimeError, AssertionError):
    use_amp = False

# Initialize scaler with correct device
scaler = GradScaler(enabled=use_amp)

# Configure logging to file and console
configure_logging()
logger = logging.getLogger(__name__)

# Setup directories
os.makedirs("app/models", exist_ok=True)
os.makedirs("app/data/logs", exist_ok=True)

# Add file handler for training logs
log_file = f"app/data/logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info(f"Logs saved to: {log_file}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device: {DEVICE}")

CHECKPOINT_PATH = "app/models/training_checkpoint.pt"
CHECKPOINT_METADATA_PATH = "app/models/training_checkpoint_metadata.json"

# EPOCHS = 5
# PATIENCE = 3
# BATCH_SIZE = 16
# LEARNING_RATE = 3e-5
#------------------------Best Hyperparameters For My PC------------------------
EPOCHS = 4
PATIENCE = 2
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
# Default values
best_f1 = 0
patience_counter = 0
start_epoch = 0
resuming_training = False
wandb_run_id = None
model = None
optimizer = None
train_loader = None
val_loader = None


def save_checkpoint_metadata(epoch, best_f1, patience_counter, wandb_run_id):
    """Save training metadata to JSON for debugging"""
    metadata = {
        "epoch": int(epoch),
        "best_f1": float(best_f1),
        "patience_counter": int(patience_counter),
        "wandb_run_id": wandb_run_id,
        "timestamp": datetime.now().isoformat(),
    }
    with open(CHECKPOINT_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.debug(f"Metadata saved: {metadata}")


def save_checkpoint(epoch, best_f1, patience_counter):
    """Save training state for resumption - with error handling"""
    try:
        checkpoint = {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_f1": float(best_f1),
            "patience_counter": int(patience_counter),
        }
        # Save to temp file first, then rename (atomic operation)
        temp_path = CHECKPOINT_PATH + ".tmp"
        torch.save(checkpoint, temp_path)
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
        os.rename(temp_path, CHECKPOINT_PATH)
        
        save_checkpoint_metadata(epoch, best_f1, patience_counter, wandb_run_id)
        logger.info(f"Checkpoint saved successfully at epoch {epoch+1}")
        return True
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return False


def load_checkpoint():
    """Load previous training state if it exists - with comprehensive error handling"""
    global best_f1, patience_counter, start_epoch, resuming_training, wandb_run_id
    
    if not os.path.exists(CHECKPOINT_PATH):
        logger.info("No checkpoint found - starting fresh training")
        return False
    
    try:
        logger.info(f"Found checkpoint file, attempting to load...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # Validate checkpoint contents
        required_keys = {"epoch", "model_state", "optimizer_state", "best_f1", "patience_counter"}
        if not all(key in checkpoint for key in required_keys):
            logger.error(f"Invalid checkpoint - missing keys. Has: {checkpoint.keys()}")
            return False
        
        # Load states
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_f1 = float(checkpoint["best_f1"])
        patience_counter = int(checkpoint["patience_counter"])
        start_epoch = int(checkpoint["epoch"]) + 1
        resuming_training = True
        
        # Try to get wandb run ID from metadata
        if os.path.exists(CHECKPOINT_METADATA_PATH):
            try:
                with open(CHECKPOINT_METADATA_PATH, "r") as f:
                    metadata = json.load(f)
                    wandb_run_id = metadata.get("wandb_run_id")
                    logger.info(f"Loaded metadata: epoch={metadata['epoch']}, best_f1={metadata['best_f1']:.4f}")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        
        logger.info("Checkpoint loaded successfully")
        logger.info(f"  Resuming from epoch {start_epoch+1}/{EPOCHS}")
        logger.info(f"  Previous best F1: {best_f1:.4f}")
        logger.info(f"  Patience counter: {patience_counter}/{PATIENCE}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        logger.warning("Starting fresh training instead")
        return False


def log_errors(texts, labels, preds):
    errors = []
    for t, l, p in zip(texts, labels, preds):
        if l != p:
            errors.append({"text": t, "true": int(l), "pred": int(p)})
    wandb.log({"errors": errors[:10]})


def evaluate():
    if not os.path.exists("app/models/best_model.pt"):
        logger.warning("No best model found for evaluation - skipping")
        return 0.0
    else:
        logger.info("Loading best model for evaluation...")
    model.eval()

    preds = []
    labels = []
    texts = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            labels.extend(y.cpu().numpy())

            # if raw text exists in dataset batch
            if "text" in batch:
                texts.extend(batch["text"])
            else:
                texts.extend([""] * len(y))

    log_confusion_matrix(labels, preds)
    log_errors(texts, labels, preds)

    f1 = f1_score(labels, preds)
    logger.debug(f"Evaluation complete - F1: {f1:.4f}")
    return f1

def main():
    global model, optimizer, train_loader, val_loader
    global best_f1, patience_counter, start_epoch, resuming_training, wandb_run_id

    # ============================================================================
    # DATASET LOADING
    # ============================================================================
    logger.info("Loading training dataset...")
    train_dataset = IMDBDataset("app/data/processed/train.csv")
    logger.info(f"Training dataset loaded: {len(train_dataset)} samples")

    logger.info("Loading validation dataset...")
    val_dataset = IMDBDataset("app/data/processed/val.csv")
    logger.info(f"Validation dataset loaded: {len(val_dataset)} samples")

    # Avoid multiprocessing bootstrap issues on Windows.
    # Windows uses 'spawn' mode which requires proper __main__ protection.
    # Using num_workers=0 is the safest approach for Windows.
    import platform
    IS_WINDOWS = platform.system() == "Windows"
    loader_workers = 0 if IS_WINDOWS or DEVICE.type == "cpu" else 2
    use_pin_memory = use_amp and DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=loader_workers,
        pin_memory=use_pin_memory
    )

    # ============================================================================
    # MODEL INITIALIZATION
    # ============================================================================
    logger.info("Loading BERT model...")
    model = load_model().to(DEVICE)
    logger.info("Model loaded successfully")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # ============================================================================
    # CHECKPOINT DETECTION & LOADING
    # ============================================================================
    logger.info("=" * 70)
    logger.info("CHECKPOINT DETECTION & LOADING")
    logger.info("=" * 70)
    load_checkpoint()

    # ============================================================================
    # WANDB INITIALIZATION (after checkpoint detection)
    # ============================================================================
    logger.info("=" * 70)
    logger.info("INITIALIZING WEIGHTS & BIASES")
    logger.info("=" * 70)

    if resuming_training and wandb_run_id:
        logger.info(f"Resuming wandb run: {wandb_run_id}")
        wandb.init(
            project="bert-imdb-sentiment",
            id=wandb_run_id,
            resume="allow",
            config={
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "model": "bert-base-uncased"
            }
        )
    else:
        logger.info("Starting new wandb run")
        run = wandb.init(
            project="bert-imdb-sentiment",
            name=f"nlp-sentiment-BERT-{datetime.now().strftime('%m%d-%H%M')}",
            config={
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "model": "bert-base-uncased"
            }
        )
        wandb_run_id = run.id
        save_checkpoint_metadata(start_epoch - 1, best_f1, patience_counter, wandb_run_id)

    logger.info(f"Wandb run ID: {wandb_run_id}")
    logger.info("=" * 70)

    for epoch in range(start_epoch, EPOCHS):
        logger.info(f"Starting epoch {epoch+1}/{EPOCHS}")
        model.train()

        total_loss = 0
        batch_count = 0

        train_progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1} Training"
        )

        for batch_idx, batch in train_progress:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.amp.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu", enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            batch_count += 1

            train_progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / batch_count:.4f}"
            })
            if batch_idx % 500 == 0:
                save_checkpoint(epoch, best_f1, patience_counter)

        avg_train_loss = total_loss / max(batch_count, 1)
        val_f1 = evaluate()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "validation_f1": val_f1
        })

        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Validation F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "app/models/best_model.pt")
            logger.info(f"New best model saved! F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.warning(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        # Always save checkpoint after each epoch for resumption
        save_checkpoint(epoch, best_f1, patience_counter)

        if patience_counter >= PATIENCE:
            logger.info("Early stopping triggered.")
            break

    wandb.finish()
    logger.info("Training complete!")

    # Clean up checkpoint file when training finishes successfully
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logger.info("Training checkpoint cleaned up")


if __name__ == "__main__":
    main()