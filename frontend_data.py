import sys
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data_preprocessing import (
    load_and_transform_md,
    load_and_transform_qd,
    aggregate_to_quarterly,
    merge_data,
)
from src.feature_selection import select_features_rlasso

from export_ar_history import build_historical_ar_csv
from export_adl_history import build_historical_adl_csv
from export_bridge_history import build_historical_bridge_csv
from frontend.components.atlanta_fed import get_historical_nowcasts
from frontend.components.atlanta_fed import annualize_gdp_growth


# @st.cache_data(ttl=3600)
# def get_actual_gdp_from_fred():
#     nowcasts_df = get_historical_nowcasts()

#     if nowcasts_df.empty or "Real GDP (Actual)" not in nowcasts_df.columns:
#         return pd.DataFrame(columns=["Year and Quarter", "Actual GDP growth"])

#     actual_df = nowcasts_df[["Real GDP (Actual)"]].reset_index()

#     # whatever the reset_index column is called (often "date"), rename it
#     first_col = actual_df.columns[0]

#     actual_df = actual_df.rename(columns={
#         first_col: "Year and Quarter",
#         "Real GDP (Actual)": "Actual GDP growth"
#     })

#     return actual_df

qd_path = ROOT_DIR / "data" / "2026-02-QD.csv"
qd_trans = load_and_transform_qd(str(qd_path))
annualized_qd = annualize_gdp_growth(qd_trans)
st.write(annualized_qd)



# @st.cache_data
# def get_modeling_data():
#     md_path = ROOT_DIR / "data" / "2026-02-MD.csv"
#     qd_path = ROOT_DIR / "data" / "2026-02-QD.csv"

#     # Step 1: monthly data
#     MD_trans = load_and_transform_md(str(md_path))

#     vars_to_drop = ['ACOGNO', 'UMCSENTx', 'TWEXAFEGSMTHx', 'ANDENOx', 'VIXCLSx']
#     MD_trans = MD_trans.drop(columns=vars_to_drop, errors='ignore')

#     # Step 2: fallback quarterly GDP from local file
#     GDP_growth = load_and_transform_qd(str(qd_path), gdp_col='GDPC1')

#     # Step 3: filter sample start
#     start_period = pd.Period('1960Q1', freq='Q')
#     start_date = start_period.start_time

#     MD_trans = MD_trans.loc[start_date:]
#     GDP_growth = GDP_growth.loc[start_period:]

#     # Step 4: aggregate + merge
#     monthly_q = aggregate_to_quarterly(MD_trans)
#     data, X, y = merge_data(monthly_q, GDP_growth)

#     # Step 5: replace GDP_growth with FRED actual GDP from get_historical_nowcasts()
#     actual_gdp_df = get_actual_gdp_from_fred().copy()

#     if not actual_gdp_df.empty:
#         fred_gdp = actual_gdp_df.rename(columns={"Actual GDP growth": "GDP_growth"}).copy()

#         # convert "2025 Q4" -> PeriodIndex("2025Q4")
#         fred_gdp["Year and Quarter"] = (
#             fred_gdp["Year and Quarter"]
#             .str.replace(" ", "", regex=False)
#         )
#         fred_gdp["Year and Quarter"] = pd.PeriodIndex(fred_gdp["Year and Quarter"], freq="Q")

#         fred_gdp = fred_gdp.set_index("Year and Quarter")[["GDP_growth"]]

#         # overwrite GDP column using FRED values where available
#         data = data.drop(columns=["GDP_growth"], errors="ignore")
#         data = data.join(fred_gdp, how="left")

#         # optional: drop rows where GDP is still missing
#         data = data.dropna(subset=["GDP_growth"])

#     # Step 6: add covid dummy
#     data['covid_dummy'] = 0
#     data.loc[
#         (data.index >= pd.Period('2020Q1', freq='Q')) &
#         (data.index <= pd.Period('2020Q2', freq='Q')),
#         'covid_dummy'
#     ] = 1

#     # Step 7: use same training split and feature selection
#     test_size = 8
#     train_data = data.iloc[:-test_size].copy()

#     selected_summary = select_features_rlasso(
#         train_data,
#         target_col='GDP_growth',
#         exclude_cols=['covid_dummy']
#     )
#     selected = list(selected_summary["feature"])

#     return data, MD_trans, selected

# actual_gdp_df = get_actual_gdp_from_fred()
# gdp_data = (
#     actual_gdp_df.assign(
#         **{
#             "Year and Quarter": lambda df: pd.PeriodIndex(
#                 df["Year and Quarter"].astype(str).str.replace(" ", "", regex=False),
#                 freq="Q"
#             )
#         }
#     )
#     .set_index("Year and Quarter")["Actual GDP growth"]
#     .sort_index()
#     .dropna()
# )
# data, md_trans, selected = get_modeling_data()

# # -- AR Data --
# @st.cache_data
# def prepare_ar_history(gdp_series):
#     output_path = ROOT_DIR / "data" / "historical_gdp_ar_predictions.csv"
#     return build_historical_ar_csv(
#         gdp_series=gdp_series,
#         output_path=output_path,
#         max_lag=8,
#         min_train_size=20,
#     )

# ar_history_df = prepare_ar_history(gdp_data)

# # -- ADL Data --
# @st.cache_data
# def prepare_adl_history(data):
#     output_path = ROOT_DIR / "data" / "historical_gdp_adl_predictions.csv"
#     return build_historical_adl_csv(
#         data=data,
#         output_path=output_path,
#         target_col="GDP_growth",
#         min_train_size=20,
#     )

# adl_history_df = prepare_adl_history(data)

# # -- Bridge Data --
# @st.cache_data
# def prepare_bridge_history(data, selected):
#     output_path = ROOT_DIR / "data" / "historical_gdp_bridge_predictions.csv"
#     return build_historical_bridge_csv(
#         data=data,
#         selected=selected,
#         output_path=output_path,
#         target_col="GDP_growth",
#         min_train_size=20,
#     )

# bridge_history_df = prepare_bridge_history(data, selected)
