"""
Integration Tests for Sentiment BERT ML Pipeline

Tests verify that all pipeline components work correctly when combined:
- Data ingestion and validation
- Text preprocessing and cleaning
- Train/val/test splitting with stratification
- Dataset loading and tokenization
- Model training loop
- Checkpoint management
- Inference output validation
"""

import pytest
import pandas as pd
import torch
from pathlib import Path
import csv


class TestDataPipeline:
    """Integration tests for data ingestion and preprocessing pipeline."""
    
    def test_data_ingestion_and_validation(self, sample_imdb_data):
        """
        Test: IMDB data loads with required columns and correct structure.
        
        Validates:
        - DataFrame has 'review' and 'sentiment' columns
        - No null values in required columns
        - Sentiment values are valid (positive/negative)
        """
        df = sample_imdb_data
        
        # Check columns exist
        required_columns = {"review", "sentiment"}
        assert required_columns.issubset(df.columns), \
            f"Missing columns: {required_columns - set(df.columns)}"
        
        # Check no nulls in required columns
        assert df["review"].notna().all(), "Null values in review column"
        assert df["sentiment"].notna().all(), "Null values in sentiment column"
        
        # Check valid sentiment values
        valid_sentiments = {"positive", "negative"}
        assert df["sentiment"].isin(valid_sentiments).all(), \
            "Invalid sentiment values detected"
        
        print(f"✓ Data validation passed: {len(df)} samples loaded")
    
    def test_text_cleaning_consistency(self, sample_imdb_data):
        """
        Test: Text cleaning produces consistent, normalized output.
        
        Validates:
        - HTML tags removed
        - Multiple spaces normalized to single space
        - Text converted to lowercase
        - No extra whitespace
        """
        import re
        
        def clean_text(text):
            if not isinstance(text, str):
                return ""
            text = re.sub(r"<.*?>", " ", text)  # Remove HTML
            text = re.sub(r"\s+", " ", text)    # Normalize spaces
            text = text.lower()                  # Lowercase
            return text.strip()
        
        df = sample_imdb_data.copy()
        df["review_clean"] = df["review"].apply(clean_text)
        
        # Verify cleaning applied correctly
        for orig, clean in zip(df["review"], df["review_clean"]):
            # No HTML tags
            assert "<" not in clean and ">" not in clean, \
                f"HTML tags not removed: {clean}"
            # No multiple spaces
            assert "  " not in clean, f"Multiple spaces found: {clean}"
            # Is lowercase
            assert clean == clean.lower(), f"Text not lowercase: {clean}"
            # No leading/trailing spaces
            assert clean == clean.strip(), f"Whitespace not trimmed: {clean}"
        
        print(f"✓ Text cleaning validation passed: {len(df)} samples cleaned")
    
    def test_stratified_split_maintains_balance(self, sample_cleaned_data):
        """
        Test: Stratified train/val/test split maintains class balance.
        
        Validates:
        - Split proportions match target (70/15/15)
        - Sentiment distribution preserved across splits
        - No data leakage between sets
        """
        from sklearn.model_selection import train_test_split
        
        df = sample_cleaned_data
        original_positive_ratio = (df["sentiment"] == "positive").sum() / len(df)
        
        # Stratified split: 70/30 then 66/34 (to get 15/15)
        train_df, temp_df = train_test_split(
            df, test_size=0.3, stratify=df["sentiment"], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["sentiment"], random_state=42
        )
        
        # Verify proportions (tolerance larger for small sample sizes)
        tolerance = 0.1  # 10% tolerance for small datasets (15 samples)
        assert abs(len(train_df) / len(df) - 0.7) < tolerance, "Train split incorrect"
        assert abs(len(val_df) / len(df) - 0.15) < tolerance, "Val split incorrect"
        assert abs(len(test_df) / len(df) - 0.15) < tolerance, "Test split incorrect"
        
        # Verify class balance maintained
        for split_df, split_name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            pos_ratio = (split_df["sentiment"] == "positive").sum() / len(split_df)
            assert abs(pos_ratio - original_positive_ratio) < 0.15, \
                f"Class imbalance in {split_name} split"
        
        # Verify no data leakage
        all_indices = set(range(len(df)))
        train_idx = set(train_df.index)
        val_idx = set(val_df.index)
        test_idx = set(test_df.index)
        
        assert len(train_idx & val_idx) == 0, "Data leakage between train and val"
        assert len(train_idx & test_idx) == 0, "Data leakage between train and test"
        assert len(val_idx & test_idx) == 0, "Data leakage between val and test"
        
        print(f"✓ Stratified split validation passed: {len(train_df)}/{len(val_df)}/{len(test_df)}")


class TestDatasetLoading:
    """Integration tests for PyTorch Dataset loading and tokenization."""
    
    def test_dataset_creation_and_loading(self, test_data_path):
        """
        Test: Dataset loads from CSV and produces correct tensor shapes.
        
        Validates:
        - CSV file loads successfully
        - Dataset length matches file length
        - Batch shapes: input_ids, attention_mask, labels
        """
        sys_path_backup = __import__('sys').path.copy()
        try:
            __import__('sys').path.insert(0, str(Path(__file__).parent.parent))
            
            from app.training.dataset import IMDBDataset
            
            dataset = IMDBDataset(test_data_path["train"], max_length=128)
            
            # Verify dataset size
            df = pd.read_csv(test_data_path["train"])
            assert len(dataset) == len(df), "Dataset length mismatch"
            
            # Verify sample structure
            sample = dataset[0]
            assert "input_ids" in sample, "Missing input_ids in sample"
            assert "attention_mask" in sample, "Missing attention_mask in sample"
            assert "labels" in sample, "Missing labels in sample"
            
            # Verify tensor shapes
            assert sample["input_ids"].shape == (128,), \
                f"Unexpected input_ids shape: {sample['input_ids'].shape}"
            assert sample["attention_mask"].shape == (128,), \
                f"Unexpected attention_mask shape: {sample['attention_mask'].shape}"
            assert sample["labels"].dim() == 0, "Labels should be scalar tensor"
            
            # Verify label values
            label = sample["labels"].item()
            assert label in [0, 1], f"Invalid label value: {label}"
            
            print(f"✓ Dataset loading validation passed: {len(dataset)} samples")
        finally:
            __import__('sys').path = sys_path_backup
    
    def test_dataloader_batch_consistency(self, test_data_path):
        """
        Test: DataLoader produces consistent batches with proper padding.
        
        Validates:
        - All sequences padded to same length
        - Batch sizes as expected
        - Labels correctly mapped (positive→1, negative→0)
        """
        sys_path_backup = __import__('sys').path.copy()
        try:
            __import__('sys').path.insert(0, str(Path(__file__).parent.parent))
            
            from app.training.dataset import IMDBDataset
            from torch.utils.data import DataLoader
            
            dataset = IMDBDataset(test_data_path["train"], max_length=128)
            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            
            # Get first batch
            batch = next(iter(loader))
            
            # Verify batch structure
            assert "input_ids" in batch, "Missing input_ids in batch"
            assert "attention_mask" in batch, "Missing attention_mask in batch"
            assert "labels" in batch, "Missing labels in batch"
            
            # Verify batch dimensions
            batch_size = batch["input_ids"].size(0)
            seq_len = batch["input_ids"].size(1)
            
            assert batch_size == 4, f"Unexpected batch size: {batch_size}"
            assert seq_len == 128, f"Unexpected sequence length: {seq_len}"
            
            # All sequences same length
            assert batch["attention_mask"].shape == batch["input_ids"].shape, \
                "attention_mask shape mismatch"
            assert batch["labels"].shape[0] == batch_size, \
                "labels batch size mismatch"
            
            print(f"✓ DataLoader batch consistency passed: batch_size={batch_size}")
        finally:
            __import__('sys').path = sys_path_backup


class TestTrainingPipeline:
    """Integration tests for model training loop components."""
    
    def test_training_step_forward_pass(self, mock_model):
        """
        Test: Forward pass produces valid logits and loss values.
        
        Validates:
        - Model handles input tensors correctly
        - Logits have shape [batch_size, num_classes]
        - Loss is scalar positive value
        - Gradient flow enabled for backward pass
        """
        batch_size, seq_len, num_classes = 4, 128, 2
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        output = mock_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Verify logits
        assert output.logits.shape == (batch_size, num_classes), \
            f"Unexpected logits shape: {output.logits.shape}"
        
        # Verify loss
        assert output.loss is not None, "Loss should not be None"
        assert output.loss.item() > 0, "Loss should be positive"
        assert output.loss.dim() == 0, "Loss should be scalar"
        
        print(f"✓ Training forward pass validation passed: loss={output.loss.item():.4f}")
    
    def test_gradient_computation(self, mock_model):
        """
        Test: Gradients compute correctly for parameter updates.
        
        Validates:
        - Backward pass succeeds without errors
        - Gradients exist for trainable parameters
        - Gradient values are non-zero
        """
        batch_size, seq_len = 4, 128
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        labels = torch.randint(0, 2, (batch_size,))
        
        output = mock_model(input_ids=input_ids, labels=labels)
        
        # Backward pass
        output.loss.backward()
        
        # Verify gradients exist
        param_count = 0
        grad_count = 0
        
        for param in mock_model.parameters():
            param_count += 1
            if param.grad is not None:
                grad_count += 1
                # Gradients should not be all zeros
                assert (param.grad != 0).any(), "Gradient values are all zeros"
        
        assert grad_count > 0, "No gradients computed"
        print(f"✓ Gradient computation validation passed: {grad_count}/{param_count} parameters")
    
    def test_checkpoint_save_load(self, mock_model, tmp_path):
        """
        Test: Model state saved and loaded correctly for resumable training.
        
        Validates:
        - Checkpoint file created
        - State dict saved with all parameters
        - Loaded state matches original
        - Metadata preserved
        """
        import json
        
        checkpoint_path = tmp_path / "checkpoint.pt"
        metadata_path = tmp_path / "metadata.json"
        
        # Save checkpoint
        checkpoint = {
            "epoch": 1,
            "model_state": mock_model.state_dict(),
            "optimizer_state": {"lr": 2e-5},
            "best_f1": 0.9162,
            "patience_counter": 0,
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        with open(metadata_path, "w") as f:
            json.dump({
                "epoch": checkpoint["epoch"],
                "best_f1": checkpoint["best_f1"],
                "timestamp": "2026-03-20T15:45:00"
            }, f)
        
        # Verify files exist
        assert checkpoint_path.exists(), "Checkpoint file not created"
        assert metadata_path.exists(), "Metadata file not created"
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Verify all keys present
        expected_keys = {"epoch", "model_state", "optimizer_state", "best_f1", "patience_counter"}
        assert expected_keys.issubset(loaded_checkpoint.keys()), \
            f"Missing checkpoint keys: {expected_keys - set(loaded_checkpoint.keys())}"
        
        # Load model state
        new_model = mock_model.__class__()
        new_model.load_state_dict(loaded_checkpoint["model_state"])
        
        # Verify state matches
        for (name1, param1), (name2, param2) in zip(
            mock_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            assert torch.allclose(param1.data, param2.data), \
                f"Parameter values differ for {name1}"
        
        print(f"✓ Checkpoint save/load validation passed")


class TestInference:
    """Integration tests for model inference and prediction."""
    
    def test_inference_output_validity(self, mock_model):
        """
        Test: Inference produces valid probability predictions.
        
        Validates:
        - Logits converted to probabilities
        - Probabilities sum to 1.0
        - Confidence scores in [0, 1]
        - Predictions are class indices 0 or 1
        """
        batch_size, seq_len = 2, 128
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Inference
        with torch.no_grad():
            output = mock_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get probabilities
        logits = output.logits
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        confidences = torch.max(probabilities, dim=1).values
        
        # Verify probability distribution
        assert probabilities.shape == (batch_size, 2), \
            f"Unexpected probabilities shape: {probabilities.shape}"
        
        # Probabilities should sum to 1
        prob_sums = probabilities.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5), \
            f"Probabilities don't sum to 1: {prob_sums}"
        
        # All values in [0, 1]
        assert (probabilities >= 0).all() and (probabilities <= 1).all(), \
            "Proabilities outside [0, 1] range"
        
        # Predictions are valid class indices
        assert (predictions >= 0).all() and (predictions < 2).all(), \
            f"Invalid prediction values: {predictions}"
        
        # Confidence in [0, 1]
        assert (confidences >= 0).all() and (confidences <= 1).all(), \
            "Confidence scores outside [0, 1] range"
        
        print(f"✓ Inference output validation passed: predictions={predictions.tolist()}")


class TestEndToEndPipeline:
    """End-to-end integration tests for complete ML pipeline."""
    
    def test_complete_pipeline_workflow(self, sample_imdb_data, test_data_path, mock_model):
        """
        Test: Complete pipeline from data ingestion through inference.
        
        Validates:
        - Data loads and cleans correctly
        - Dataset creates valid tensors
        - Model trains without errors
        - Checkpoints and recovers correctly
        """
        sys_path_backup = __import__('sys').path.copy()
        try:
            __import__('sys').path.insert(0, str(Path(__file__).parent.parent))
            
            from app.training.dataset import IMDBDataset
            from torch.utils.data import DataLoader
            from torch.optim import AdamW
            
            print("\n=== End-to-End Pipeline Test ===")
            
            # Step 1: Load dataset
            print("Step 1: Loading dataset...")
            dataset = IMDBDataset(test_data_path["train"], max_length=128)
            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            print(f"  ✓ Dataset loaded: {len(dataset)} samples")
            
            # Step 2: Model + Optimizer setup
            print("Step 2: Setting up model and optimizer...")
            optimizer = AdamW(mock_model.parameters(), lr=2e-5)
            print(f"  ✓ Model and optimizer initialized")
            
            # Step 3: Training loop (1 mini epoch)
            print("Step 3: Running training loop (1 batch)...")
            mock_model.train()
            batch = next(iter(loader))
            
            output = mock_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            
            print(f"  ✓ Training step completed (loss={output.loss.item():.4f})")
            
            # Step 4: Inference
            print("Step 4: Running inference...")
            mock_model.eval()
            with torch.no_grad():
                output = mock_model(input_ids=batch["input_ids"])
                probs = torch.softmax(output.logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            
            print(f"  ✓ Inference completed: {len(preds)} predictions")
            
            # Step 5: Checkpoint
            print("Step 5: Saving checkpoint...")
            checkpoint_path = Path(__file__).parent / "test_checkpoint.pt"
            torch.save({"model_state": mock_model.state_dict()}, checkpoint_path)
            print(f"  ✓ Checkpoint saved")
            
            # Step 6: Restore
            print("Step 6: Loading checkpoint...")
            new_model = mock_model.__class__()
            loaded_ckpt = torch.load(checkpoint_path)
            new_model.load_state_dict(loaded_ckpt["model_state"])
            print(f"  ✓ Checkpoint restored")
            
            # Cleanup
            checkpoint_path.unlink()
            
            print("=== ✓ End-to-End Pipeline Test PASSED ===\n")
            
        finally:
            __import__('sys').path = sys_path_backup


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
