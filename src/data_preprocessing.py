"""
Nowcasting project – Data preprocessing
------------------------------------------------------------
This script:
1. Loads FRED‑MD (monthly) and FRED‑QD (quarterly) CSV files.
2. Applies the recommended t‑code transformations to achieve stationarity.
3. Aggregates monthly indicators to quarterly frequency.
4. Merges with quarterly GDP growth.
"""

from config import *

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def load_and_transform_md(filepath):
    """
    Load FRED‑MD CSV, extract transformation codes, apply them,
    and return a DataFrame of stationary monthly series (index = date).
    """
    md = pd.read_csv(filepath)
    # First row contains t‑codes
    tcodes = md.iloc[0]
    md = md.iloc[1:].copy()
    
    # Parse dates
    md['sasdate'] = pd.to_datetime(md['sasdate'], format='%m/%d/%Y')
    md.set_index('sasdate', inplace=True)
    
    # Apply transformations to every column
    transformed_list = []
    for col in md.columns:
        code = int(tcodes[col])
        s = transform_series(md[col].astype(float), code)
        s.name = col
        transformed_list.append(s)
    
    md_trans = pd.concat(transformed_list, axis=1)
    #md_trans.dropna(inplace=True)          # should not remove all rows with NaN values 
    print("FRED‑MD transformation complete. Shape:", md_trans.shape)
    return md_trans


def load_and_transform_qd(filepath, gdp_col='GDPC1'):
    """
    Load FRED‑QD CSV, extract transformation codes, apply them to the GDP series,
    and return a Series of quarterly GDP growth (index = period).
    """
    qd = pd.read_csv(filepath)
    tcodes_qd = qd.iloc[1]     # FRED-QD stores metadata/header rows before actual observations.
    qd = qd.iloc[2:].copy()    # Row 1 contains transformation codes for the variables.

    if gdp_col not in qd.columns:
        raise ValueError(f"{gdp_col} not found in quarterly dataset.")

    # Parse dates (format may vary; adjust if needed)
    qd['sasdate'] = pd.to_datetime(qd['sasdate'], format='%m/%d/%Y', errors='coerce')
    qd.dropna(subset=['sasdate'], inplace=True)
    qd.set_index('sasdate', inplace=True)
    
    # Transform the GDP column
    code_gdp = int(tcodes_qd[gdp_col])
    gdp_raw = qd[gdp_col].astype(float)
    gdp_trans = transform_series(gdp_raw, code_gdp)
    gdp_trans.name = 'GDP_growth'
    gdp_trans.dropna(inplace=True)
    
    # Convert index to quarterly period
    gdp_trans.index = gdp_trans.index.to_period('Q')
    print("FRED‑QD transformation complete. Length:", len(gdp_trans))
    return gdp_trans


def transform_series(series, code):
    """
    Apply a given transformation code to a pandas Series.
    Codes follow McCracken & Ng (2016):
        1 : no transformation
        2 : first difference
        3 : second difference
        4 : log
        5 : log first difference
        6 : log second difference
    """
    if code == 1:
        return series
    elif code == 2:
        return series.diff()
    elif code == 3:
        return series.diff().diff()
    elif code == 4:
        return np.log(series.where(series > 0))
    elif code == 5:
        return np.log(series.where(series > 0)).diff()
    elif code == 6:
        return np.log(series.where(series > 0)).diff().diff()
    else:
        return series   # fallback (should not happen)


def aggregate_to_quarterly(monthly_df):
    """
    Convert monthly DataFrame (index = date) to quarterly by averaging.
    Returns a DataFrame with a PeriodIndex (quarters).
    """
    # Add a quarter column
    monthly_df = monthly_df.copy()
    monthly_df['quarter'] = monthly_df.index.to_period('Q')
    # For now, all monthly indicators are aggregated using quarterly averages.
    quarterly = monthly_df.groupby('quarter').mean()
    return quarterly


def merge_data(monthly_q, gdp_series):
    """
    Merge quarterly aggregates of monthly indicators with GDP growth.
    Returns a DataFrame (rows = quarters) and separate X, y arrays.
    """
    data = monthly_q.join(gdp_series, how='inner').dropna()
    X = data.drop(columns=['GDP_growth']).values
    y = data['GDP_growth'].values
    print("Quarterly dataset shape:", data.shape)
    print("Number of predictors:", X.shape[1])
    return data, X, y

## Add covid dummy variable 
def add_covid_dummy(data, start='2020Q1', end='2020Q4'):
    data = data.copy()
    start_q = pd.Period(start, freq='Q')
    end_q = pd.Period(end, freq='Q')
    data['covid_dummy'] = ((data.index >= start_q) & (data.index <= end_q)).astype(int)
    return data

## Add one function that prepares the full training dataframe
def prepare_training_data(md_path, qd_path, gdp_col='GDPC1', add_covid=False):
    md_trans = load_and_transform_md(md_path)
    gdp_growth = load_and_transform_qd(qd_path, gdp_col=gdp_col)
    monthly_q = aggregate_to_quarterly(md_trans)
    data = merge_data(monthly_q, gdp_growth)

    if add_covid:
        data = add_covid_dummy(data)

    return md_trans, monthly_q, data

