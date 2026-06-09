## GLOBAL CONFIGURATION & DATA DICTIONARIES for Fraud dataset

COLUMN_NAMES = [
    "user_id",
    "signup_time",
    "purchase_time",
    "purchase_value",
    "device_id",
    "source",
    "browser",
    "sex",
    "age",
    "ip_address",
    "class"
]


NUMERICAL_COLS = [
    "user_id",
    "purchase_value",
    "age",
    "ip_address",
    "class"
]

CATEGORICAL_COLS = [
    "signup_time",
    "purchase_time",
    "device_id",
    "source",
    "browser",
    "sex"
]

TARGET_COL = "class"