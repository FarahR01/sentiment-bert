from sklearn.model_selection import train_test_split
import logging

def stratified_split(df, label_col="sentiment", test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets with stratification.
    """
    # Split train + temp (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), stratify=df[label_col], random_state=random_state
    )
    # Split temp into validation and test
    val_relative_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_relative_size, stratify=temp_df[label_col], random_state=random_state
    )
    logging.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df