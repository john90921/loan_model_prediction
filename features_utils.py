import numpy as np
def clean_features(df):
    df = df.copy()

    # Replace pl`aceholder values with NaN
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Replace negative values with NaN for certain columns
    missing_cols = [
    "OWN_CAR_AGE",
    "OCCUPATION_TYPE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    ]

    for col in missing_cols:
        df[f"{col}_MISSING"] = df[col].isna().astype(int);
    df = df.drop(columns = "SK_ID_CURR")
    df = df[df['CODE_GENDER'] != 'XNA']
    return df

def engineer_features(df):
    df = df.copy()
 
    # Age and employment
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    employed = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["EMPLOYMENT_YEARS"] = -employed / 365.25
    df["EMPLOYMENT_ANOMALY"] = (
        df["DAYS_EMPLOYED"] == 365243
    ).astype(int)

    # Financial ratios
    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] /
        df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    df["ANNUITY_CREDIT_RATIO"] = (
        df["AMT_ANNUITY"] /
        df["AMT_CREDIT"].replace(0, np.nan)
    )

    # External scores
    ext_cols = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]

    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_COUNT"] = df[ext_cols].notna().sum(axis=1)
    text_cols = df.select_dtypes(include=["object"]).columns
    df[text_cols] = df[text_cols].astype("category")
    
    return df.replace([np.inf, -np.inf], np.nan)