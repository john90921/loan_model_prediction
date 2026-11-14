import pandas as pd
import numpy as np
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
    df = df.drop(columns="SK_ID_CURR")
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

    # def engineer_features(df):
    df = df.copy()

    # =====================================
    # 1. FINANCIAL BURDEN FEATURES
    # =====================================

    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
        df["CREDIT_TERM_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(
            0, np.nan
        )

    if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(df.columns):
        df["INCOME_CREDIT_DIFF"] = df["AMT_INCOME_TOTAL"] - df["AMT_CREDIT"]

    if {"AMT_INCOME_TOTAL", "AMT_ANNUITY"}.issubset(df.columns):
        df["ANNUITY_INCOME_DIFF"] = df["AMT_INCOME_TOTAL"] - df["AMT_ANNUITY"]

    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(df.columns):
        df["LOAN_TO_GOODS_GAP"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]

    # =====================================
    # 2. FAMILY FEATURES
    # =====================================

    if {"AMT_INCOME_TOTAL", "CNT_CHILDREN"}.issubset(df.columns):
        df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"] + 1)

    if {"AMT_CREDIT", "CNT_CHILDREN"}.issubset(df.columns):
        df["CREDIT_PER_CHILD"] = df["AMT_CREDIT"] / (df["CNT_CHILDREN"] + 1)

    if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["ADULTS_ESTIMATE"] = df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]

        df["CHILDREN_FAMILY_RATIO"] = df["CNT_CHILDREN"] / df[
            "CNT_FAM_MEMBERS"
        ].replace(0, np.nan)

    # =====================================
    # 3. AGE AND EMPLOYMENT FEATURES
    # =====================================

    if {"DAYS_BIRTH", "DAYS_EMPLOYED"}.issubset(df.columns):
        age_days = -df["DAYS_BIRTH"].replace(0, np.nan)

        employed_days = df["DAYS_EMPLOYED"].replace(365243, np.nan)

        df["EMPLOYMENT_AGE_RATIO"] = employed_days / age_days

    if {"DAYS_BIRTH", "DAYS_REGISTRATION"}.issubset(df.columns):
        df["REGISTRATION_AGE_RATIO"] = df["DAYS_REGISTRATION"] / df[
            "DAYS_BIRTH"
        ].replace(0, np.nan)

    if {"DAYS_BIRTH", "DAYS_ID_PUBLISH"}.issubset(df.columns):
        df["ID_CHANGE_AGE_RATIO"] = df["DAYS_ID_PUBLISH"] / df["DAYS_BIRTH"].replace(
            0, np.nan
        )

    # =====================================
    # 4. EXTERNAL SCORE INTERACTIONS
    # =====================================

    ext_cols = [
        c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns
    ]

    if len(ext_cols) >= 2:
        scores = df[ext_cols]

        df["EXT_SOURCE_STD"] = scores.std(axis=1)
        df["EXT_SOURCE_PRODUCT"] = scores.prod(axis=1, min_count=len(ext_cols))
        df["TOTAL_EXT_MISSING"] = scores.isna().sum(axis=1)

        # Difference between available external scores
        df["EXT_SOURCE_RANGE"] = scores.max(axis=1) - scores.min(axis=1)

    if {"EXT_SOURCE_1", "EXT_SOURCE_2"}.issubset(df.columns):
        df["EXT_1_2_PRODUCT"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]

    if {"EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(df.columns):
        df["EXT_2_3_PRODUCT"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

    # =====================================
    # 5. OWNERSHIP FEATURES
    # =====================================

    if "FLAG_OWN_CAR" in df.columns:
        df["HAS_CAR"] = (df["FLAG_OWN_CAR"] == "Y").astype("int8")

    if "FLAG_OWN_REALTY" in df.columns:
        df["HAS_REALTY"] = (df["FLAG_OWN_REALTY"] == "Y").astype("int8")

    if {"FLAG_OWN_CAR", "FLAG_OWN_REALTY"}.issubset(df.columns):
        df["CAR_AND_REALTY"] = (
            (df["FLAG_OWN_CAR"] == "Y") & (df["FLAG_OWN_REALTY"] == "Y")
        ).astype("int8")

    # =====================================
    # 6. MISSINGNESS FEATURES
    # =====================================

    missing_cols = [
        "OWN_CAR_AGE",
        "OCCUPATION_TYPE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_GOODS_PRICE",
        "AMT_ANNUITY",
    ]

    for col in missing_cols:
        if col in df.columns:
            df[col + "_MISSING"] = df[col].isna().astype("int8")

    # =====================================
    # 7. FINAL CLEANING
    # =====================================

    return df.replace([np.inf, -np.inf], np.nan)


# def engineer_features(df):
#     df = df.copy()

#     # Age and employment
#     df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

#     employed = df["DAYS_EMPLOYED"].replace(365243, np.nan)
#     df["EMPLOYMENT_YEARS"] = -employed / 365.25
#     df["EMPLOYMENT_ANOMALY"] = (
#         df["DAYS_EMPLOYED"] == 365243
#     ).astype(int)

#     # Financial ratios
#     df["CREDIT_INCOME_RATIO"] = (
#         df["AMT_CREDIT"] /
#         df["AMT_INCOME_TOTAL"].replace(0, np.nan)
#     )

#     df["ANNUITY_INCOME_RATIO"] = (
#         df["AMT_ANNUITY"] /
#         df["AMT_INCOME_TOTAL"].replace(0, np.nan)
#     )

#     df["ANNUITY_CREDIT_RATIO"] = (
#         df["AMT_ANNUITY"] /
#         df["AMT_CREDIT"].replace(0, np.nan)
#     )

#     # External scores
#     ext_cols = [
#         "EXT_SOURCE_1",
#         "EXT_SOURCE_2",
#         "EXT_SOURCE_3"
#     ]

#     df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
#     df["EXT_SOURCE_COUNT"] = df[ext_cols].notna().sum(axis=1)
#     text_cols = df.select_dtypes(include=["object"]).columns
#     df[text_cols] = df[text_cols].astype("category")

#     return df.replace([np.inf, -np.inf], np.nan)

# def engineer_features2(df):
#     df = df.copy()

#     # Financial ratios
#     ratios = [
#         ("CREDIT_ANNUITY_RATIO", "AMT_CREDIT", "AMT_ANNUITY"),
#         ("GOODS_CREDIT_RATIO", "AMT_GOODS_PRICE", "AMT_CREDIT"),
#         ("INCOME_PER_FAMILY", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
#         ("CREDIT_PER_FAMILY", "AMT_CREDIT", "CNT_FAM_MEMBERS"),
#     ]

#     for new_col, numerator, denominator in ratios:
#         if numerator in df.columns and denominator in df.columns:
#             den = df[denominator].replace(0, np.nan)
#             df[new_col] = df[numerator] / den

#     # Age
#     if "DAYS_BIRTH" in df.columns:
#         df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

#     # Employment: handle the known placeholder
#     if "DAYS_EMPLOYED" in df.columns:
#         days = df["DAYS_EMPLOYED"]
#         df["EMPLOYED_ANOMALY"] = (days == 365243).astype("int8")
#         valid_days = days.mask(days == 365243)
#         df["EMPLOYMENT_YEARS"] = -valid_days / 365.25

#         # if "DAYS_BIRTH" in df.columns:
#         #     age_days = -df["DAYS_BIRTH"].replace(0, np.nan)
#         #     df["EMPLOYMENT_AGE_RATIO"] = valid_days / age_days

#     # External scores
#     ext = [
#         c for c in
#         ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
#         if c in df.columns
#     ]

#     if ext:
#         scores = df[ext]
#         df["EXT_MEAN"] = scores.mean(axis=1)
#         df["EXT_MIN"] = scores.min(axis=1)
#         df["EXT_MAX"] = scores.max(axis=1)
#         df["EXT_SPREAD"] = scores.max(axis=1) - scores.min(axis=1)
#         df["EXT_COUNT"] = scores.notna().sum(axis=1)

#         # for col in ext:
#         #     df[col + "_MISSING"] = df[col].isna().astype("int8")

#     # Social-circle indicators
#     for col in [
#         "DEF_30_CNT_SOCIAL_CIRCLE",
#         "DEF_60_CNT_SOCIAL_CIRCLE",
#         "OBS_30_CNT_SOCIAL_CIRCLE",
#         "OBS_60_CNT_SOCIAL_CIRCLE",
#     ]:
#         if col in df.columns:
#             df[col + "_MISSING"] = df[col].isna().astype("int8")

#     # Count document flags, if available
#     doc_cols = [
#         c for c in df.columns
#         if c.startswith("FLAG_DOCUMENT_")
#     ]
#     if doc_cols:
#         df["DOCUMENT_FLAGS_SUM"] = df[doc_cols].fillna(0).sum(axis=1)

#     # Preserve missingness in selected variables
#     for col in ["OWN_CAR_AGE", "OCCUPATION_TYPE", "AMT_GOODS_PRICE"]:
#         if col in df.columns:
#             df[col + "_MISSING"] = df[col].isna().astype("int8")

#     return df.replace([np.inf, -np.inf], np.nan)
