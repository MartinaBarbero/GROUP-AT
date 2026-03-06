"""
SAAM Project - Part I, Section 1: Data Cleaning
Group AT: North America (AMER) / Scope 1+2
This script:
  1. Loads all raw Datastream files
  2. Removes error rows and firms with no matching data
  3. Filters to the assigned region (AMER)
  4. Handles low prices (RI < 0.5 → NaN)
  5. Computes monthly simple returns
  6. Detects delisted firms (return = -100%)
  7. Handles missing values in carbon/revenue data (forward fill)
  8. Applies the stale price filter
  9. Prints diagnostics and saves cleaned data to pickle files

At the end, you have clean DataFrames ready for portfolio construction.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION — Adjust these if needed
# =============================================================================
REGION = 'AMER'               # North America
SCOPE = '1+2'                 # Scope 1 + Scope 2
LOW_PRICE_THRESHOLD = 0.5     # RI below this → treated as missing
STALE_THRESHOLD = 0.50        # If >50% of returns are 0 over 10y window → exclude
MIN_RETURN_MONTHS = 36        # At least 3 years of monthly returns required
ESTIMATION_WINDOW = 120       # 10 years × 12 months
Y0 = 2013                    # First allocation year
Y_END = 2024                 # Last allocation year

# Path to data files (put all xlsx in the same folder)
DATA_DIR = './'

# =============================================================================
# STEP 1: LOAD RAW DATA
# =============================================================================
print("=" * 70)
print("STEP 1: Loading raw Datastream files")
print("=" * 70)

static       = pd.read_excel(f'{DATA_DIR}Static_2025.xlsx')
ri_m_raw     = pd.read_excel(f'{DATA_DIR}DS_RI_T_USD_M_2025.xlsx')
mv_m_raw     = pd.read_excel(f'{DATA_DIR}DS_MV_T_USD_M_2025.xlsx')
mv_y_raw     = pd.read_excel(f'{DATA_DIR}DS_MV_T_USD_Y_2025.xlsx')
ri_y_raw     = pd.read_excel(f'{DATA_DIR}DS_RI_T_USD_Y_2025.xlsx')
co2_s1_raw   = pd.read_excel(f'{DATA_DIR}DS_CO2_SCOPE_1_Y_2025.xlsx')
co2_s2_raw   = pd.read_excel(f'{DATA_DIR}DS_CO2_SCOPE_2_Y_2025.xlsx')
rev_raw      = pd.read_excel(f'{DATA_DIR}DS_REV_Y_2025.xlsx')
rf_raw       = pd.read_excel(f'{DATA_DIR}Risk_Free_Rate_2025.xlsx')

print(f"  Static file:        {static.shape[0]} firms")
print(f"  RI monthly:         {ri_m_raw.shape[0]} rows × {ri_m_raw.shape[1]} cols")
print(f"  MV monthly:         {mv_m_raw.shape[0]} rows × {mv_m_raw.shape[1]} cols")
print(f"  MV yearly:          {mv_y_raw.shape[0]} rows × {mv_y_raw.shape[1]} cols")
print(f"  CO2 Scope 1:        {co2_s1_raw.shape[0]} rows × {co2_s1_raw.shape[1]} cols")
print(f"  CO2 Scope 2:        {co2_s2_raw.shape[0]} rows × {co2_s2_raw.shape[1]} cols")
print(f"  Revenue:            {rev_raw.shape[0]} rows × {rev_raw.shape[1]} cols")
print(f"  Risk-free rate:     {rf_raw.shape[0]} months")

# =============================================================================
# STEP 2: REMOVE ERROR ROWS / MISSING ISINs
# =============================================================================
# Datastream sometimes inserts an error row (first row) with "$$ER: ..." 
# and rows where ISIN is NaN (no matching firm found).
# We remove these from ALL datasets.
print("\n" + "=" * 70)
print("STEP 2: Removing Datastream error rows and missing ISINs")
print("=" * 70)

def clean_monthly(df_raw):
    """Clean monthly Datastream file: drop error/NaN ISIN rows, set ISIN as index."""
    df = df_raw.copy()
    n_before = len(df)
    # Drop rows with missing ISIN
    df = df[df['ISIN'].notna()]
    # Drop rows where NAME contains Datastream error strings
    if df['NAME'].dtype == object:
        df = df[~df['NAME'].astype(str).str.contains('\\$\\$ER', na=False)]
    n_after = len(df)
    print(f"    Removed {n_before - n_after} error/missing rows → {n_after} firms remain")
    df = df.set_index('ISIN')
    name_col = df.pop('NAME')  # keep NAME separately
    # Convert all price/MV columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    return df, name_col

def clean_annual(df_raw, label=""):
    """Clean annual Datastream file: drop error/NaN ISIN rows, set ISIN as index."""
    df = df_raw.copy()
    n_before = len(df)
    df = df[df['ISIN'].notna()]
    # Check for error strings in any column
    for col in df.columns:
        if df[col].dtype == object and col not in ['NAME', 'ISIN']:
            df = df[~df[col].astype(str).str.contains('\\$\\$ER', na=False)]
            break
    n_after = len(df)
    print(f"  {label}: removed {n_before - n_after} error rows → {n_after} firms")
    df = df.set_index('ISIN')
    df = df.drop(columns=['NAME'], errors='ignore')
    # Ensure column names are integers (years)
    df.columns = [int(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

print("  Monthly RI:")
ri_m, names_ri = clean_monthly(ri_m_raw)
print("  Monthly MV:")
mv_m, _ = clean_monthly(mv_m_raw)
co2_s1 = clean_annual(co2_s1_raw, "CO2 Scope 1")
co2_s2 = clean_annual(co2_s2_raw, "CO2 Scope 2")
rev    = clean_annual(rev_raw,     "Revenue")
mv_y   = clean_annual(mv_y_raw,   "MV yearly")

# Save date columns for later reference
dates_m = ri_m.columns.tolist()       # monthly dates (datetime)
print(f"\n  Monthly date range: {dates_m[0].strftime('%Y-%m')} to {dates_m[-1].strftime('%Y-%m')}")
print(f"  Total monthly dates: {len(dates_m)}")

# =============================================================================
# STEP 3: FILTER TO ASSIGNED REGION (AMER)
# =============================================================================
print("\n" + "=" * 70)
print(f"STEP 3: Filtering to region = {REGION}")
print("=" * 70)

region_isins = set(static[static['Region'] == REGION]['ISIN'].values)
print(f"  Firms in Static with Region={REGION}: {len(region_isins)}")

# Keep only ISINs present in ALL datasets (intersection)
all_datasets_isins = (
    region_isins
    & set(ri_m.index)
    & set(mv_m.index)
    & set(co2_s1.index)
    & set(co2_s2.index)
    & set(rev.index)
    & set(mv_y.index)
)
common_isins = sorted(all_datasets_isins)
print(f"  Firms in {REGION} present in ALL files: {len(common_isins)}")

# Differences (for diagnostics)
only_in_static = region_isins - set(ri_m.index)
if only_in_static:
    print(f"  ⚠ {len(only_in_static)} AMER firms in Static but missing from RI monthly")

# Apply filter
ri_m   = ri_m.loc[common_isins]
mv_m   = mv_m.loc[common_isins]
co2_s1 = co2_s1.loc[common_isins]
co2_s2 = co2_s2.loc[common_isins]
rev    = rev.loc[common_isins]
mv_y   = mv_y.loc[common_isins]

print(f"  All datasets now have {len(common_isins)} firms")

# =============================================================================
# STEP 4: LOW PRICE FILTER (RI < 0.5 → NaN)
# =============================================================================
# "I suggest treating all prices below 0.5 as missing values."
# This prevents extreme/infinite returns from near-zero prices.
print("\n" + "=" * 70)
print(f"STEP 4: Low price filter (RI < {LOW_PRICE_THRESHOLD} → NaN)")
print("=" * 70)

n_low_before = (ri_m < LOW_PRICE_THRESHOLD).sum().sum()
# Count how many are exactly 0
n_zero = (ri_m == 0).sum().sum()

ri_m[ri_m < LOW_PRICE_THRESHOLD] = np.nan

print(f"  Prices set to NaN: {n_low_before} cells (of which {n_zero} were exactly 0)")
print(f"  Total NaN in RI after filter: {ri_m.isna().sum().sum()}")

# =============================================================================
# STEP 5: COMPUTE MONTHLY SIMPLE RETURNS
# =============================================================================
# R_{i,t} = P_{i,t} / P_{i,t-1} - 1
print("\n" + "=" * 70)
print("STEP 5: Computing monthly simple returns")
print("=" * 70)

returns_m = ri_m.pct_change(axis=1)
# The first column has no return (it's the base price Dec 1999), drop it
returns_m = returns_m.iloc[:, 1:]
dates_ret = returns_m.columns.tolist()

print(f"  Returns matrix: {returns_m.shape[0]} firms × {len(dates_ret)} months")
print(f"  Return date range: {dates_ret[0].strftime('%Y-%m')} to {dates_ret[-1].strftime('%Y-%m')}")

# Quick sanity check: look for extreme returns
extreme_mask = returns_m.abs() > 5  # returns > 500%
n_extreme = extreme_mask.sum().sum()
print(f"  Extreme returns (|R| > 500%): {n_extreme}")
if n_extreme > 0:
    print("    → These may come from price jumps after missing data; "
          "they will be handled by the delisting logic below.")

# =============================================================================
# STEP 6: HANDLE DELISTED FIRMS (return = -100%)
# =============================================================================
# "When a firm is delisted, the price goes to 0 → realized return of -100%."
# Detection: RI goes from valid (≥0.5) to NaN and stays NaN permanently.
print("\n" + "=" * 70)
print("STEP 6: Detecting delisted firms (return = -100% at delisting)")
print("=" * 70)

delisted_firms = []
n_delisting = 0

for isin in ri_m.index:
    prices = ri_m.loc[isin].values
    for j in range(1, len(prices)):
        # Previous month had a valid price, current month is NaN
        if (pd.notna(prices[j - 1]) and prices[j - 1] >= LOW_PRICE_THRESHOLD
                and pd.isna(prices[j])):
            # Check all subsequent months are also NaN (permanent delisting)
            if all(pd.isna(prices[j:])):
                # Set the return at the delisting month to -100%
                ret_col = dates_ret[j - 1]  # returns_m has 1 fewer col than ri_m
                returns_m.loc[isin, ret_col] = -1.0
                n_delisting += 1
                # Record for diagnostics
                delisted_firms.append({
                    'ISIN': isin,
                    'Name': names_ri.get(isin, 'N/A') if isin in names_ri.index else 'N/A',
                    'Last_Price_Date': dates_m[j - 1].strftime('%Y-%m'),
                    'Delisting_Date': dates_m[j].strftime('%Y-%m'),
                })
                break

print(f"  Delisting events detected: {n_delisting}")
if delisted_firms:
    df_del = pd.DataFrame(delisted_firms)
    print(f"\n  First 10 delisted firms:")
    print(df_del.head(10).to_string(index=False))

# =============================================================================
# STEP 7: HANDLE MISSING VALUES IN CARBON & REVENUE (FORWARD FILL)
# =============================================================================
# "When the missing value is between two available years or at the end of the sample,
#  just use the number from the previous year."
# "When the missing value is at the beginning of the sample, you cannot invest in
#  this firm until numbers are made available."
print("\n" + "=" * 70)
print("STEP 7: Forward-filling carbon and revenue data")
print("=" * 70)

def describe_missing(df, name):
    total = df.size
    missing = df.isna().sum().sum()
    pct = 100 * missing / total if total > 0 else 0
    print(f"  {name}: {missing}/{total} missing ({pct:.1f}%)")

print("  BEFORE forward fill:")
describe_missing(co2_s1, "CO2 Scope 1")
describe_missing(co2_s2, "CO2 Scope 2")
describe_missing(rev,    "Revenue")

# Forward fill along columns (years)
co2_s1 = co2_s1.ffill(axis=1)
co2_s2 = co2_s2.ffill(axis=1)
rev    = rev.ffill(axis=1)

print("\n  AFTER forward fill:")
describe_missing(co2_s1, "CO2 Scope 1")
describe_missing(co2_s2, "CO2 Scope 2")
describe_missing(rev,    "Revenue")

# Compute total CO2 (Scope 1 + Scope 2) since we are Group AT (Scope 1+2)
co2_total = co2_s1.add(co2_s2, fill_value=0)
# If both scopes are NaN for a firm-year, co2_total remains NaN
both_nan = co2_s1.isna() & co2_s2.isna()
co2_total[both_nan] = np.nan
print(f"\n  CO2 Total (Scope 1+2) computed: {co2_total.shape}")
describe_missing(co2_total, "CO2 Total")

# =============================================================================
# STEP 8: STALE PRICE FILTER
# =============================================================================
# "If the proportion of months with return = 0 exceeds 50%, exclude the firm."
# This is checked per year using the 10-year estimation window.
print("\n" + "=" * 70)
print("STEP 8: Stale price filter diagnostic")
print("=" * 70)

# Build a helper: December index in dates_ret for each year
dec_ret_indices = {}
for i, d in enumerate(dates_ret):
    if d.month == 12:
        dec_ret_indices[d.year] = i

dec_price_indices = {}
for i, d in enumerate(dates_m):
    if d.month == 12:
        dec_price_indices[d.year] = i

# Show how many firms would be excluded by the stale filter each year
print(f"  Stale threshold: {STALE_THRESHOLD * 100:.0f}% zero returns over 10y window\n")
for Y in range(Y0, Y_END + 1):
    dec_idx = dec_ret_indices[Y]
    start_idx = max(0, dec_idx - ESTIMATION_WINDOW + 1)
    window_cols = dates_ret[start_idx:dec_idx + 1]
    ret_window = returns_m[window_cols]

    n_stale = 0
    for isin in ret_window.index:
        firm_ret = ret_window.loc[isin].dropna()
        if len(firm_ret) > 0:
            zero_pct = (firm_ret == 0).sum() / len(firm_ret)
            if zero_pct > STALE_THRESHOLD:
                n_stale += 1
    print(f"  Year {Y}: {n_stale} firms excluded by stale filter")

# =============================================================================
# STEP 9: BUILD INVESTMENT SET FUNCTION
# =============================================================================
# This function combines all the cleaning criteria to define the investable
# universe at the end of each year Y.
print("\n" + "=" * 70)
print("STEP 9: Investment set construction (all criteria combined)")
print("=" * 70)

def get_investment_set(year):
    """
    At end of year Y, define investable firms satisfying:
      1. RI available at end of year Y (price not NaN)
      2. >= MIN_RETURN_MONTHS valid returns in the 10y window
      3. Zero-return proportion <= STALE_THRESHOLD
      4. CO2 total (Scope 1+2) available at end of year Y
    
    Returns:
      eligible_isins: list of ISINs in the investment set
      window_cols:    the return columns for the 10y estimation window
      exclusion_reasons: dict with counts of excluded firms by reason
    """
    dec_idx = dec_ret_indices[year]
    start_idx = max(0, dec_idx - ESTIMATION_WINDOW + 1)
    window_cols = dates_ret[start_idx:dec_idx + 1]
    ret_window = returns_m[window_cols]

    eligible = []
    reasons = {'no_price': 0, 'few_returns': 0, 'stale': 0, 'no_carbon': 0}

    for isin in ret_window.index:
        # Criterion 1: RI available at end of year Y
        price_eoy = ri_m.loc[isin, dates_m[dec_price_indices[year]]]
        if pd.isna(price_eoy):
            reasons['no_price'] += 1
            continue

        # Criterion 2: Sufficient return observations
        firm_ret = ret_window.loc[isin].dropna()
        if len(firm_ret) < MIN_RETURN_MONTHS:
            reasons['few_returns'] += 1
            continue

        # Criterion 3: Stale price filter
        zero_pct = (firm_ret == 0).sum() / len(firm_ret)
        if zero_pct > STALE_THRESHOLD:
            reasons['stale'] += 1
            continue

        # Criterion 4: CO2 (Scope 1+2) data available
        has_co2 = (year in co2_total.columns
                   and pd.notna(co2_total.loc[isin, year]))
        if not has_co2:
            reasons['no_carbon'] += 1
            continue

        eligible.append(isin)

    return eligible, window_cols, reasons


# Run it for each year and report
print(f"\n  {'Year':<6} {'Eligible':>10} {'No Price':>10} {'Few Ret':>10} "
      f"{'Stale':>10} {'No Carbon':>10}")
print("  " + "-" * 60)

investment_sets = {}
for Y in range(Y0, Y_END + 1):
    eligible, window_cols, reasons = get_investment_set(Y)
    investment_sets[Y] = eligible
    print(f"  {Y:<6} {len(eligible):>10} {reasons['no_price']:>10} "
          f"{reasons['few_returns']:>10} {reasons['stale']:>10} "
          f"{reasons['no_carbon']:>10}")

# =============================================================================
# STEP 10: PREPARE RISK-FREE RATE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: Preparing risk-free rate")
print("=" * 70)

rf_df = rf_raw.copy()
rf_df.columns = ['date_code', 'RF']
rf_df['year'] = rf_df['date_code'] // 100
rf_df['month'] = rf_df['date_code'] % 100
# RF appears to be annualized % → convert to monthly decimal
rf_df['RF_monthly'] = rf_df['RF'] / 100 / 12
rf_dict = {(int(r['year']), int(r['month'])): r['RF_monthly']
           for _, r in rf_df.iterrows()}
print(f"  RF available for {len(rf_dict)} months")
print(f"  Sample: Jan 2014 RF (monthly) = {rf_dict.get((2014, 1), 'N/A'):.6f}")

# =============================================================================
# STEP 11: SAVE CLEANED DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 11: Saving cleaned data")
print("=" * 70)

cleaned = {
    'ri_m': ri_m,
    'mv_m': mv_m,
    'mv_y': mv_y,
    'returns_m': returns_m,
    'co2_s1': co2_s1,
    'co2_s2': co2_s2,
    'co2_total': co2_total,
    'rev': rev,
    'static': static,
    'rf_dict': rf_dict,
    'dates_m': dates_m,
    'dates_ret': dates_ret,
    'dec_ret_indices': dec_ret_indices,
    'dec_price_indices': dec_price_indices,
    'investment_sets': investment_sets,
    'common_isins': common_isins,
}

pd.to_pickle(cleaned, 'cleaned_data.pkl')
print("  All cleaned data saved to: cleaned_data.pkl")
print("  Load it later with: cleaned = pd.read_pickle('cleaned_data.pkl')")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("DATA CLEANING COMPLETE — SUMMARY")
print("=" * 70)
print(f"""
  Region:                     {REGION} (North America)
  Scope:                      {SCOPE}
  Total firms in Static:      {static.shape[0]}
  Firms in {REGION}:              {len(region_isins)}
  Firms with data in all files: {len(common_isins)}
  Low price threshold:        {LOW_PRICE_THRESHOLD}
  Stale price threshold:      {STALE_THRESHOLD * 100:.0f}%
  Min return months required: {MIN_RETURN_MONTHS}
  Delisting events:           {n_delisting}
  
  Investment set sizes:
""")
for Y in range(Y0, Y_END + 1):
    print(f"    {Y}: {len(investment_sets[Y])} firms")

print(f"""
  Cleaned data saved to: cleaned_data.pkl
  
  Next step: Load cleaned_data.pkl and run portfolio optimization (Section 2).
""")
