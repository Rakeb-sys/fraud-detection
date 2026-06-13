import numpy as np
import pandas as pd
import pytest


from src.preprocessing import FeatureEngineerTransform, DataFrameLabelEncoder, scale_features, evaluate

SEED = 42

@pytest.fixture
def sample_df():
    np.random.seed(SEED)
    n_samples = 200
    
    # 1. Generate the Fraud Class (3% fraud rate)
    is_fraud = np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])
    
    # 2. Generate Base Timestamps
    base_dates = pd.to_datetime(
        np.random.randint(
            pd.Timestamp("2015-01-01").value,
            pd.Timestamp("2015-07-01").value,
            size=n_samples,
        )
    )
    
    # 3. Correlate behaviors based on fraud class
    # Fraudsters have much shorter signup-to-purchase times and higher sharing counts
    signup_to_purchase = np.where(
        is_fraud == 1,
        np.random.exponential(scale=100, size=n_samples),        # Very fast (seconds/minutes)
        np.random.exponential(scale=1_000_000, size=n_samples), # Normal (days/weeks)
    ).round(4)
    
    sharing_counts = np.where(
        is_fraud == 1,
        np.random.randint(5, 15, size=n_samples),   # Highly shared devices/IPs
        np.random.randint(1, 3, size=n_samples),    # Unique devices/IPs
    )
    
    # 4. Assemble the realistic dataset
    df = pd.DataFrame({
        "user_id": np.random.randint(1000, 500000, size=n_samples),
        "signup_time": base_dates,
        "purchase_value": np.random.randint(10, 150, size=n_samples),
        "device_id": ["".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 13)) for _ in range(n_samples)],
        "source": np.random.choice(["SEO", "Ads", "Direct"], size=n_samples, p=[0.4, 0.4, 0.2]),
        "browser": np.random.choice(["Chrome", "Opera", "Safari", "IE", "FireFox"], size=n_samples),
        "sex": np.random.choice(["M", "F"], size=n_samples),
        "age": np.random.randint(18, 70, size=n_samples),
        "ip_address": np.random.uniform(10_000_000, 3_000_000_000, size=n_samples).round(4),
        "class": is_fraud,
        "country": np.random.choice(["United States", "Japan", "China", "Canada"], size=n_samples),
        "signup_to_purchase_time": signup_to_purchase,
        "device_sharing_count": sharing_counts,
        "ip_sharing_count": sharing_counts,
    })
    
    # 5. Derive calculated dependent columns
    df["purchase_time"] = df["signup_time"] + pd.to_timedelta(df["signup_to_purchase_time"], unit="s")
    df["ip_int"] = df["ip_address"].astype(int)
    df["immediate_purchase"] = (df["signup_to_purchase_time"] < 5).astype(int) # 1 if under 5 seconds
    
    # Reorder columns to exactly match your list
    column_order = [
        "user_id", "signup_time", "purchase_time", "purchase_value", "device_id", 
        "source", "browser", "sex", "age", "ip_address", "class", "ip_int", 
        "country", "signup_to_purchase_time", "immediate_purchase", 
        "device_sharing_count", "ip_sharing_count"
    ]
    
    return df[column_order]


class Test_feature_engineer:
    def test_new_columns_creation(self,sample_df):
        processed_df = FeatureEngineerTransform(sample_df)

        expected_cols = [
            "signup_hour", "signup_dayofweek", "signup_month","signup_year",
            "purchase_hour", "purchase_dayofweek", "purchase_month","purchase_year",
            "signup_to_purchase_time","immediate_purchase"
        ]

        for col in expected_cols:
            assert col in processed_df.columns

    def test_shape_has_more_columns(self, sample_df):
        result = FeatureEngineerTransform(sample_df)
        assert result.shape[1] > sample_df.shape[1]
