"""
Home Credit application-table feature pipeline (all feature sets combined).

Contents
  1. clean_features + original add_* feature groups + create_selected_features
  2. add_extra_features (V2 groups)
  3. V3 features (credit / timeline / external / bureau / social / region /
     group stats / frequency encoding)
  4. build_all_features(): one call for train (+ optional test)

Quick start
    X_train, y_train, X_test = build_all_features(train_raw, test_raw)
"""

import numpy as np
import pandas as pd


# ============================================================================
# 1. ORIGINAL PIPELINE
# ============================================================================
def clean_features(df):
    df = df.copy()

    # Replace pl`aceholder values with NaN
    # df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # # Replace negative values with NaN for certain columns
    # missing_cols = [
    # "OCCUPATION_TYPE",
    # "EXT_SOURCE_2",
    # "EXT_SOURCE_3",
    # ]

    # for col in missing_cols:
    #     df[f"{col}_MISSING"] = df[col].isna().astype(int);
    text_cols = df.select_dtypes(include=["object"]).columns
    df[text_cols] = df[text_cols].astype("category")
    df = df[df["CODE_GENDER"] != "XNA"]
    return df


def add_financial_features(df):
    df = df.copy()

    # Loan and income ratios
    df["CREDIT_TO_ANNUITY"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
    df["GOODS_TO_INCOME"] = df["AMT_GOODS_PRICE"] / df["AMT_INCOME_TOTAL"].replace(
        0, np.nan
    )
    df["ANNUITY_TO_CREDIT"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)
    df["CREDIT_MINUS_INCOME"] = df["AMT_CREDIT"] - df["AMT_INCOME_TOTAL"]
    df["ANNUITY_MINUS_INCOME"] = df["AMT_ANNUITY"] - df["AMT_INCOME_TOTAL"]

    # Household affordability
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(
        0, np.nan
    )
    df["CREDIT_PER_PERSON"] = df["AMT_CREDIT"] / df["CNT_FAM_MEMBERS"].replace(
        0, np.nan
    )
    df["ANNUITY_PER_PERSON"] = df["AMT_ANNUITY"] / df["CNT_FAM_MEMBERS"].replace(
        0, np.nan
    )

    # Goods financing gap
    df["CREDIT_GOODS_GAP"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    df["CREDIT_GOODS_GAP_RATIO"] = df["CREDIT_GOODS_GAP"] / df["AMT_CREDIT"].replace(
        0, np.nan
    )

    # Log transformations for highly skewed variables
    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        df[f"LOG_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


def add_age_employment_features(df):
    df = df.copy()

    age_days = df["DAYS_BIRTH"].abs()
    employed_days = df["DAYS_EMPLOYED"].replace(365243, np.nan).abs()

    df["AGE_YEARS_NEW"] = age_days / 365.25
    df["EMPLOYMENT_YEARS_NEW"] = employed_days / 365.25

    df["EMPLOYMENT_TO_AGE"] = employed_days / age_days.replace(0, np.nan)
    df["YEARS_NOT_EMPLOYED"] = (age_days - employed_days) / 365.25

    df["REGISTRATION_YEARS"] = df["DAYS_REGISTRATION"].abs() / 365.25
    df["ID_PUBLISH_YEARS"] = df["DAYS_ID_PUBLISH"].abs() / 365.25
    df["PHONE_CHANGE_YEARS"] = df["DAYS_LAST_PHONE_CHANGE"].abs() / 365.25

    df["REGISTRATION_TO_AGE"] = df["DAYS_REGISTRATION"].abs() / age_days.replace(
        0, np.nan
    )
    df["ID_PUBLISH_TO_AGE"] = df["DAYS_ID_PUBLISH"].abs() / age_days.replace(0, np.nan)
    df["PHONE_CHANGE_TO_AGE"] = df["DAYS_LAST_PHONE_CHANGE"].abs() / age_days.replace(
        0, np.nan
    )

    # Employment anomaly must be calculated BEFORE replacing 365243
    df["EMPLOYMENT_ANOMALY_NEW"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")

    return df


def add_external_score_features(df):
    df = df.copy()

    cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    available = [c for c in cols if c in df.columns]

    if not available:
        return df

    scores = df[available]

    df["EXT_MEAN_NEW"] = scores.mean(axis=1)
    df["EXT_MIN_NEW"] = scores.min(axis=1)
    df["EXT_MAX_NEW"] = scores.max(axis=1)
    df["EXT_STD_NEW"] = scores.std(axis=1)
    df["EXT_RANGE_NEW"] = df["EXT_MAX_NEW"] - df["EXT_MIN_NEW"]
    df["EXT_MISSING_COUNT_NEW"] = scores.isna().sum(axis=1)

    # Products
    df["EXT_PRODUCT_NEW"] = scores.prod(axis=1, min_count=len(available))

    # Pairwise differences
    if all(c in df.columns for c in cols):
        df["EXT_DIFF_12"] = df["EXT_SOURCE_1"] - df["EXT_SOURCE_2"]
        df["EXT_DIFF_13"] = df["EXT_SOURCE_1"] - df["EXT_SOURCE_3"]
        df["EXT_DIFF_23"] = df["EXT_SOURCE_2"] - df["EXT_SOURCE_3"]

        df["EXT_PRODUCT_12"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
        df["EXT_PRODUCT_13"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
        df["EXT_PRODUCT_23"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

    return df


def add_bureau_request_features(df):
    df = df.copy()

    cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]

    cols = [c for c in cols if c in df.columns]

    if cols:
        df["BUREAU_REQUEST_TOTAL"] = df[cols].sum(axis=1, min_count=1)
        df["BUREAU_REQUEST_MISSING"] = df[cols].isna().sum(axis=1)
        df["BUREAU_REQUEST_NONZERO"] = (df[cols].fillna(0) > 0).sum(axis=1)

        if "AMT_REQ_CREDIT_BUREAU_YEAR" in df.columns:
            df["BUREAU_YEAR_REQUEST"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"]

        if "AMT_REQ_CREDIT_BUREAU_MON" in df.columns:
            df["BUREAU_MONTH_REQUEST"] = df["AMT_REQ_CREDIT_BUREAU_MON"]

    return df


def add_asset_features(df):
    df = df.copy()

    if "FLAG_OWN_CAR" in df.columns and "FLAG_OWN_REALTY" in df.columns:
        df["OWNS_CAR_AND_PROPERTY"] = (
            (df["FLAG_OWN_CAR"] == "Y") & (df["FLAG_OWN_REALTY"] == "Y")
        ).astype("int8")

        df["OWNS_NEITHER"] = (
            (df["FLAG_OWN_CAR"] == "N") & (df["FLAG_OWN_REALTY"] == "N")
        ).astype("int8")

    if "OWN_CAR_AGE" in df.columns:
        df["CAR_AGE_MISSING"] = df["OWN_CAR_AGE"].isna().astype("int8")
        df["CAR_AGE_TO_CLIENT_AGE"] = df["OWN_CAR_AGE"] / (
            df["DAYS_BIRTH"].abs() / 365.25
        ).replace(0, np.nan)

    return df


def add_missing_document_features(df):
    df = df.copy()

    document_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]

    if document_cols:
        df["DOCUMENT_COUNT"] = df[document_cols].sum(axis=1)
        df["DOCUMENT_MISSING_COUNT"] = df[document_cols].isna().sum(axis=1)

    # General missing-value count
    df["MISSING_FEATURE_COUNT"] = df.isna().sum(axis=1)

    important_cols = [
        "OCCUPATION_TYPE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_GOODS_PRICE",
        "AMT_ANNUITY",
        "OWN_CAR_AGE",
    ]

    for col in important_cols:
        if col in df.columns:
            df[f"{col}_IS_MISSING"] = df[col].isna().astype("int8")

    return df


def add_social_features(df):
    df = df.copy()

    pairs = [
        ("OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE"),
        ("OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"),
    ]

    for obs, default in pairs:
        if obs in df.columns and default in df.columns:
            df[f"{default}_RATIO"] = df[default] / df[obs].replace(0, np.nan)

            df[f"{default}_TO_OBS_GAP"] = df[obs] - df[default]

    return df


def add_category_interactions(df):
    df = df.copy()

    pairs = [
        ("NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"),
        ("NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"),
        ("OCCUPATION_TYPE", "ORGANIZATION_TYPE"),
        ("CODE_GENDER", "NAME_INCOME_TYPE"),
        ("NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE"),
    ]

    for col1, col2 in pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f"{col1}_{col2}_INTERACTION"] = (
                df[col1].astype("string").fillna("Missing")
                + "_"
                + df[col2].astype("string").fillna("Missing")
            ).astype("category")

    return df


# ============================================================================
# 2. V2 EXTRA FEATURES
# ============================================================================
def add_extra_features(df, groups=None):
    df = df.copy()

    if groups is None:
        groups = [
            "finance",
            "family",
            "contact",
            "region",
            "timing",
            "housing",
            "external",
        ]

    def num(col):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return None

    def ratio(a, b):
        return a / b.replace(0, np.nan)

    # =========================================
    # 1. ADDITIONAL FINANCIAL AFFORDABILITY
    # =========================================
    if "finance" in groups:
        income = num("AMT_INCOME_TOTAL")
        credit = num("AMT_CREDIT")
        annuity = num("AMT_ANNUITY")
        goods = num("AMT_GOODS_PRICE")

        if income is not None and credit is not None:
            df["V2_CREDIT_TO_INCOME"] = ratio(credit, income)

        if income is not None and annuity is not None:
            df["V2_ANNUITY_TO_INCOME"] = ratio(annuity, income)
            df["V2_INCOME_AFTER_ANNUITY_RATIO"] = ratio(income - annuity, income)

        if credit is not None and goods is not None:
            df["V2_CREDIT_TO_GOODS"] = ratio(credit, goods)
            df["V2_DOWN_PAYMENT_RATIO"] = ratio(goods - credit, goods)

        if annuity is not None and goods is not None:
            df["V2_ANNUITY_TO_GOODS"] = ratio(annuity, goods)

    # =========================================
    # 2. FAMILY DEPENDENCY FEATURES
    # =========================================
    if "family" in groups:
        children = num("CNT_CHILDREN")
        family = num("CNT_FAM_MEMBERS")

        if children is not None and family is not None:
            adults = family - children

            df["V2_ADULTS_ESTIMATE"] = adults
            df["V2_CHILDREN_PER_ADULT"] = ratio(children, adults.where(adults > 0))
            df["V2_IS_SINGLE_ADULT_HOUSEHOLD"] = ((adults == 1) & (family <= 2)).astype(
                "int8"
            )

        if "NAME_FAMILY_STATUS" in df.columns:
            status = df["NAME_FAMILY_STATUS"].astype("string")
            df["V2_IS_SINGLE_STATUS"] = (
                status.str.lower().str.contains("single", na=False)
            ).astype("int8")

    # =========================================
    # 3. CONTACT AND COMMUNICATION FEATURES
    # =========================================
    if "contact" in groups:
        contact_cols = ["FLAG_MOBIL", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
        contact_cols = [c for c in contact_cols if c in df.columns]

        if contact_cols:
            contact_data = df[contact_cols].apply(pd.to_numeric, errors="coerce")

            df["V2_CONTACT_CHANNEL_COUNT"] = contact_data.fillna(0).sum(axis=1)
            df["V2_CONTACT_FLAGS_MISSING"] = contact_data.isna().sum(axis=1)
            df["V2_NO_CONTACT_CHANNEL"] = (
                contact_data.fillna(0).sum(axis=1) == 0
            ).astype("int8")

        work_cols = ["FLAG_EMP_PHONE", "FLAG_WORK_PHONE"]
        work_cols = [c for c in work_cols if c in df.columns]

        if work_cols:
            work_data = df[work_cols].apply(pd.to_numeric, errors="coerce")
            df["V2_WORK_PHONE_FLAG_COUNT"] = work_data.fillna(0).sum(axis=1)

    # =========================================
    # 4. REGIONAL MOBILITY FEATURES
    # =========================================
    if "region" in groups:
        region_cols = [
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION",
        ]

        city_cols = [
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "LIVE_CITY_NOT_WORK_CITY",
        ]

        region_cols = [c for c in region_cols if c in df.columns]
        city_cols = [c for c in city_cols if c in df.columns]

        if region_cols:
            region_data = df[region_cols].apply(pd.to_numeric, errors="coerce")
            df["V2_REGION_MISMATCH_COUNT"] = region_data.fillna(0).sum(axis=1)

        if city_cols:
            city_data = df[city_cols].apply(pd.to_numeric, errors="coerce")
            df["V2_CITY_MISMATCH_COUNT"] = city_data.fillna(0).sum(axis=1)

        all_mismatch = region_cols + city_cols
        if all_mismatch:
            mismatch_data = df[all_mismatch].apply(pd.to_numeric, errors="coerce")
            df["V2_TOTAL_LOCATION_MISMATCH"] = mismatch_data.fillna(0).sum(axis=1)

    # =========================================
    # 5. APPLICATION TIMING FEATURES
    # =========================================
    if "timing" in groups:
        hour = num("HOUR_APPR_PROCESS_START")

        if hour is not None:
            df["V2_APPLICATION_HOUR_SIN"] = np.sin(2 * np.pi * hour / 24)
            df["V2_APPLICATION_HOUR_COS"] = np.cos(2 * np.pi * hour / 24)

            df["V2_IS_MORNING_APPLICATION"] = (hour.between(6, 11)).astype("int8")

            df["V2_IS_EVENING_APPLICATION"] = (hour.between(17, 23)).astype("int8")

        if "WEEKDAY_APPR_PROCESS_START" in df.columns:
            weekday = df["WEEKDAY_APPR_PROCESS_START"].astype("string").str.lower()

            weekday_map = {
                "monday": 0,
                "tuesday": 1,
                "wednesday": 2,
                "thursday": 3,
                "friday": 4,
                "saturday": 5,
                "sunday": 6,
            }

            weekday_num = weekday.map(weekday_map)

            df["V2_WEEKDAY_NUMBER"] = weekday_num
            df["V2_IS_WEEKEND_APPLICATION"] = (weekday_num >= 5).astype("int8")

            df["V2_WEEKDAY_SIN"] = np.sin(2 * np.pi * weekday_num / 7)
            df["V2_WEEKDAY_COS"] = np.cos(2 * np.pi * weekday_num / 7)

    # =========================================
    # 6. PROPERTY / HOUSING FEATURES
    # =========================================
    if "housing" in groups:
        property_bases = [
            "APARTMENTS",
            "BASEMENTAREA",
            "YEARS_BEGINEXPLUATATION",
            "YEARS_BUILD",
            "COMMONAREA",
            "ELEVATORS",
            "ENTRANCES",
            "FLOORSMAX",
            "FLOORSMIN",
            "LANDAREA",
            "LIVINGAPARTMENTS",
            "LIVINGAREA",
            "NONLIVINGAPARTMENTS",
            "NONLIVINGAREA",
            "TOTALAREA",
        ]

        suffixes = ["AVG", "MODE", "MEDI"]
        housing_cols = []

        for base in property_bases:
            for suffix in suffixes:
                col = f"{base}_{suffix}"
                if col in df.columns:
                    housing_cols.append(col)

        # Aggregate property measurements by suffix
        for suffix in suffixes:
            cols = [
                f"{base}_{suffix}"
                for base in property_bases
                if f"{base}_{suffix}" in df.columns
            ]

            if cols:
                values = df[cols].apply(pd.to_numeric, errors="coerce")

                df[f"V2_HOUSING_{suffix}_MEAN"] = values.mean(axis=1)
                df[f"V2_HOUSING_{suffix}_STD"] = values.std(axis=1)
                df[f"V2_HOUSING_{suffix}_MISSING"] = values.isna().sum(axis=1)

        # Difference between AVG, MODE and MEDI measurements
        dispersion_count = 0

        for base in property_bases:
            cols = [
                f"{base}_{suffix}"
                for suffix in suffixes
                if f"{base}_{suffix}" in df.columns
            ]

            if len(cols) >= 2:
                values = df[cols].apply(pd.to_numeric, errors="coerce")

                df[f"V2_{base}_PROPERTY_SPREAD"] = values.max(axis=1) - values.min(
                    axis=1
                )
                dispersion_count += 1

        if housing_cols:
            housing_data = df[housing_cols]
            df["V2_TOTAL_HOUSING_MISSING"] = housing_data.isna().sum(axis=1)
            df["V2_HOUSING_AVAILABLE_COUNT"] = housing_data.notna().sum(axis=1)

    # =========================================
    # 7. EXTERNAL SCORE INTERACTIONS
    # =========================================
    if "external" in groups:
        age_days = num("DAYS_BIRTH")
        age_years = age_days.abs() / 365.25 if age_days is not None else None

        credit_income = None
        if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            credit_income = ratio(num("AMT_CREDIT"), num("AMT_INCOME_TOTAL"))

        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            score = num(col)

            if score is not None:
                # Non-linear score effect
                df[f"V2_{col}_SQUARED"] = score**2

                # Interaction with applicant age
                if age_years is not None:
                    df[f"V2_{col}_AGE"] = score * age_years

                # Interaction with credit burden
                if credit_income is not None:
                    df[f"V2_{col}_CREDIT_BURDEN"] = score * credit_income

    return df.replace([np.inf, -np.inf], np.nan)


def create_selected_features(df, selected=None):
    if selected is None:
        selected = [
            "financial",
            "age",
            "external",
            "bureau",
            "assets",
            "missing",
            "social",
            "categorical",
        ]

    df = df.copy()

    feature_functions = {
        "financial": add_financial_features,
        "age": add_age_employment_features,
        "external": add_external_score_features,
        "bureau": add_bureau_request_features,
        "assets": add_asset_features,
        "missing": add_missing_document_features,
        "social": add_social_features,
        "categorical": add_category_interactions,
    }

    for group in selected:
        if group in feature_functions:
            df = feature_functions[group](df)

    return df.replace([np.inf, -np.inf], np.nan)


# ============================================================================
# 3. V3 FEATURES
# ============================================================================
EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
BUREAU_COLS = [
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

# Starting guess only (EXT_SOURCE_3 / 2 are usually stronger than 1). Tune it, or
# fit a logistic regression on the three scores and paste the coefficients here.
DEFAULT_EXT_WEIGHTS = {"EXT_SOURCE_1": 1.0, "EXT_SOURCE_2": 1.5, "EXT_SOURCE_3": 2.0}

GROUP_KEYS = [
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "REGION_RATING_CLIENT_W_CITY",
]
GROUP_VALUES = ["income", "credit", "annuity", "goods", "ext_mean", "emp_years"]

ALL_GROUPS = [
    "credit",
    "timeline",
    "external",
    "bureau",
    "social",
    "region",
    "group",
    "frequency",
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _num(df, col):
    """Numeric Series for `col`, or all-NaN if the column does not exist."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _div(a, b):
    """a / b with 0 -> NaN in the denominator."""
    return a / b.where(b != 0)


def _key(df, col):
    """Group key as string, NaN -> 'Missing'."""
    if col not in df.columns:
        return pd.Series("Missing", index=df.index, dtype="string")
    return df[col].astype("string").fillna("Missing")


def _base(df):
    """Shared numeric building blocks (computed once)."""
    b = pd.DataFrame(index=df.index)
    b["income"] = _num(df, "AMT_INCOME_TOTAL")
    b["credit"] = _num(df, "AMT_CREDIT")
    b["annuity"] = _num(df, "AMT_ANNUITY")
    b["goods"] = _num(df, "AMT_GOODS_PRICE")
    b["children"] = _num(df, "CNT_CHILDREN")
    b["family"] = _num(df, "CNT_FAM_MEMBERS")

    b["age_years"] = _num(df, "DAYS_BIRTH").abs() / 365.25
    # 365243 is the "not employed / pensioner" placeholder
    b["emp_years"] = _num(df, "DAYS_EMPLOYED").replace(365243, np.nan).abs() / 365.25
    b["reg_years"] = _num(df, "DAYS_REGISTRATION").abs() / 365.25
    b["id_years"] = _num(df, "DAYS_ID_PUBLISH").abs() / 365.25
    b["phone_years"] = _num(df, "DAYS_LAST_PHONE_CHANGE").abs() / 365.25

    b["term"] = _div(b["credit"], b["annuity"])  # loan term in payment periods

    for i, c in enumerate(EXT_COLS, 1):
        b[f"ext{i}"] = _num(df, c)
    b["ext_mean"] = b[["ext1", "ext2", "ext3"]].mean(axis=1)
    return b


def _weighted_ext(b, weights):
    """NaN-aware weighted mean of the available EXT_SOURCE scores."""
    w = np.array([weights.get(c, 1.0) for c in EXT_COLS], dtype=float)
    vals = b[["ext1", "ext2", "ext3"]].to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    wsum = (mask * w).sum(axis=1)
    num = np.where(mask, vals * w, 0.0).sum(axis=1)
    out = np.where(wsum > 0, num / np.where(wsum > 0, wsum, 1.0), np.nan)
    return pd.Series(out, index=b.index)


# ----------------------------------------------------------------------------
# feature blocks (each returns a dict of Series)
# ----------------------------------------------------------------------------
def _credit_block(b):
    f = {}
    monthly_income = b["income"] / 12  # AMT_INCOME_TOTAL is yearly, annuity is per period

    f["V3_AGE_AT_MATURITY"] = b["age_years"] + b["term"] / 12
    f["V3_DISPOSABLE_INCOME_MONTHLY"] = monthly_income - b["annuity"]
    f["V3_DISPOSABLE_PER_PERSON"] = _div(monthly_income - b["annuity"], b["family"])
    f["V3_INCOME_PER_CHILD"] = b["income"] / (1 + b["children"])
    f["V3_CREDIT_PER_EMPLOYED_YEAR"] = b["credit"] / (1 + b["emp_years"])
    f["V3_FINANCED_EXTRA_TO_INCOME"] = _div(b["credit"] - b["goods"], b["income"])
    f["V3_TERM_YEARS_TO_AGE"] = _div(b["term"] / 12, b["age_years"])
    return f


def _timeline_block(b):
    f = {}
    f["V3_AGE_AT_ID_PUBLISH"] = b["age_years"] - b["id_years"]
    f["V3_AGE_AT_REGISTRATION"] = b["age_years"] - b["reg_years"]
    f["V3_REG_MINUS_ID_YEARS"] = b["reg_years"] - b["id_years"]
    f["V3_PHONE_MINUS_ID_YEARS"] = b["phone_years"] - b["id_years"]
    f["V3_EMPLOYED_MINUS_REG_YEARS"] = b["emp_years"] - b["reg_years"]
    f["V3_EMPLOYED_MINUS_ID_YEARS"] = b["emp_years"] - b["id_years"]
    return f


def _external_block(b, weights):
    f = {}
    ext_w = _weighted_ext(b, weights)
    f["V3_EXT_WEIGHTED"] = ext_w
    f["V3_EXT_GEOMEAN"] = np.exp(np.log(b[["ext1", "ext2", "ext3"]].clip(lower=1e-6)).mean(axis=1))

    f["V3_EXT_W_X_TERM"] = ext_w * b["term"]
    f["V3_EXT_W_X_EMP_YEARS"] = ext_w * b["emp_years"]
    f["V3_EXT_W_X_INCOME_PP"] = ext_w * np.log1p(_div(b["income"], b["family"]).clip(lower=0))
    f["V3_EXT_W_X_INCOME_TO_CREDIT"] = ext_w * _div(b["income"], b["credit"])

    for i in (1, 2, 3):
        f[f"V3_EXT{i}_X_TERM"] = b[f"ext{i}"] * b["term"]
        f[f"V3_EXT{i}_X_EMP_YEARS"] = b[f"ext{i}"] * b["emp_years"]
    return f


def _bureau_block(df):
    f = {}
    v = {c: _num(df, c) for c in BUREAU_COLS}
    h, d, w, m, q, y = (v[c] for c in BUREAU_COLS)

    recent = pd.concat([h, d, w, m], axis=1).sum(axis=1, min_count=1)
    total = pd.concat([h, d, w, m, q, y], axis=1).sum(axis=1, min_count=1)

    # The windows are non-overlapping; approximate lengths: MON~1, QRT~2, YEAR~9 months
    rate_recent = recent
    rate_qrt = q / 2
    rate_year = y / 9

    f["V3_BUREAU_RECENT_COUNT"] = recent
    f["V3_BUREAU_RECENT_SHARE"] = recent / (total + 1)
    f["V3_BUREAU_ACCEL_SHORT"] = rate_recent - rate_qrt
    f["V3_BUREAU_ACCEL_LONG"] = rate_qrt - rate_year
    return f


def _social_block(df):
    f = {}
    o30 = _num(df, "OBS_30_CNT_SOCIAL_CIRCLE")
    d30 = _num(df, "DEF_30_CNT_SOCIAL_CIRCLE")
    o60 = _num(df, "OBS_60_CNT_SOCIAL_CIRCLE")
    d60 = _num(df, "DEF_60_CNT_SOCIAL_CIRCLE")

    f["V3_DEF_NEW_30_60"] = d60 - d30  # defaults that appeared in days 31-60
    f["V3_OBS_NEW_30_60"] = o60 - o30
    # Laplace-smoothed rates keep a value when OBS == 0 (the plain ratio gives NaN)
    f["V3_DEF30_SMOOTH"] = d30 / (o30 + 1)
    f["V3_DEF60_SMOOTH"] = d60 / (o60 + 1)
    f["V3_DEF_SMOOTH_DIFF"] = f["V3_DEF60_SMOOTH"] - f["V3_DEF30_SMOOTH"]
    return f


def _region_block(df, b):
    f = {}
    r = _num(df, "REGION_RATING_CLIENT")
    rc = _num(df, "REGION_RATING_CLIENT_W_CITY")
    f["V3_REGION_RATING_DIFF"] = rc - r
    f["V3_REGION_RATING_SUM"] = rc + r
    f["V3_REGION_RATING_X_EXT"] = rc * b["ext_mean"]
    return f


# ----------------------------------------------------------------------------
# reference-based blocks (group stats, frequency encoding)
# ----------------------------------------------------------------------------
def fit_v3_reference(df, group_keys=None, group_values=None, freq_cols=None):
    """
    Learn group means and category frequencies. Target-free, so it is safe to fit
    on train+test together. Fit ONCE, then reuse for every split.
    """
    group_keys = [k for k in (group_keys or GROUP_KEYS) if k in df.columns]
    group_values = group_values or GROUP_VALUES
    if freq_cols is None:
        freq_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    b = _base(df)
    group_means = {}
    for key in group_keys:
        group_means[key] = b[group_values].groupby(_key(df, key)).mean()

    freq_maps = {
        c: _key(df, c).value_counts(normalize=True) for c in freq_cols if c in df.columns
    }
    return {
        "group_keys": group_keys,
        "group_values": group_values,
        "group_means": group_means,
        "freq_maps": freq_maps,
    }


def _group_block(df, b, ref):
    f = {}
    for key in ref["group_keys"]:
        k = _key(df, key)
        for val in ref["group_values"]:
            mean = k.map(ref["group_means"][key][val]).astype(float)
            name = f"V3_GRP_{key}_{val.upper()}"
            f[f"{name}_DIFF"] = b[val] - mean
            f[f"{name}_RATIO"] = _div(b[val], mean)
    return f


def _frequency_block(df, ref):
    f = {}
    for col, freq in ref["freq_maps"].items():
        f[f"V3_FREQ_{col}"] = _key(df, col).map(freq).astype(float).fillna(0.0)
    return f


# ----------------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------------
def add_v3_features(df, ref=None, groups=None, ext_weights=None):
    groups = ALL_GROUPS if groups is None else groups
    ext_weights = ext_weights or DEFAULT_EXT_WEIGHTS

    if ("group" in groups or "frequency" in groups) and ref is None:
        raise ValueError("'group' / 'frequency' blocks need ref=fit_v3_reference(...)")

    b = _base(df)
    new = {}

    if "credit" in groups:
        new.update(_credit_block(b))
    if "timeline" in groups:
        new.update(_timeline_block(b))
    if "external" in groups:
        new.update(_external_block(b, ext_weights))
    if "bureau" in groups:
        new.update(_bureau_block(df))
    if "social" in groups:
        new.update(_social_block(df))
    if "region" in groups:
        new.update(_region_block(df, b))
    if "group" in groups:
        new.update(_group_block(df, b, ref))
    if "frequency" in groups:
        new.update(_frequency_block(df, ref))

    new_df = pd.DataFrame(new, index=df.index).replace([np.inf, -np.inf], np.nan)
    new_df = new_df.astype("float32")
    return pd.concat([df, new_df], axis=1)


# ============================================================================
# 4. ONE-CALL PIPELINE
# ============================================================================
def build_all_features(
    data,
    selected=None,
    extra_groups=None,
    v3_groups=None,
    ext_weights=None,
    target_col="TARGET",
):
    """
    clean -> original groups -> V2 extras -> V3 features.

    `data` may be train only, or train+test concatenated (test TARGET = NaN).
    TARGET is held out while features are built, then reattached, so it cannot
    change MISSING_FEATURE_COUNT. V3 group/frequency stats are fit on `data`
    itself (no target used). Returns one DataFrame with the original index
    reset to 0..n-1; rows with CODE_GENDER == "XNA" are dropped.
    """
    data = data.reset_index(drop=True)
    target = data[target_col] if target_col in data.columns else None
    full = data.drop(columns=[target_col], errors="ignore")

    full = clean_features(full)
    full = create_selected_features(full, selected)
    full = add_extra_features(full, extra_groups)

    ref = fit_v3_reference(full)
    full = add_v3_features(full, ref, groups=v3_groups, ext_weights=ext_weights)

    if target is not None:
        full[target_col] = target.loc[full.index]
    return full