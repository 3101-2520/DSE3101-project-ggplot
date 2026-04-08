# DSE3101 gg(plot) 

![Dashboard Preview](frontend/assets/Dashboard_preview.png)

[Dashboard (Streamlit App)](https://dse3101-proj-nncf8h3qkwgetp9nj6vzbc.streamlit.app/)

## Research Topic

**Nowcasting the economy:** This project serves to provide the nowcast for US GDP growth, as well as benchmark comparisons with industry models such as Atlanta's GDPNOW and St Louis Fed, through a clean and concise dashboard that is updated in real-time.

## Methodology

1. **Data preprocessing**
2. **Feature selection using LASSO**
3. **Bridge regression**
4. **Benchmark AR/ ADL/ RF models**
5. **Model Evaluation**
6. **Live Nowcasting**

## Dataset

- **FRED-MD**: Monthly macroeconomic dataset from the Federal Reserve bank of St. Louis
- **FRED-QD**: Quarterly macroeconomic dataset including real GDP
- **FRED-API**: Used for live nowcasting with the latest available monthly indicators and quarterly GDP

---

## Project Structure

```text
DSE3101-Proj/
├── .streamlit/                 # Streamlit theme and configuration
├── data/                       # Raw datasets, model predictions, and live API CSVs
├── figures/                    # Generated plots and visualizations
├── frontend/                   # Streamlit dashboard modules
│   ├── assets/                 # Static assets and images
│   ├── components/             # UI elements (biz cycle, live graphs, config panels)
│   ├── main.py                 # Frontend main layout logic and main entry point to run the Streamlit app
│   ├── utils.py                # Frontend helper functions
│   └── export_*.py             # Scripts for exporting historical data
├── models/                     # Saved model artifacts and weights
├── src/                        # Backend Data Science & ML Pipelines
│   ├── api_preprocessing.py    # Fetches and cleans live FRED data
│   ├── data_preprocessing.py   # Processes historical FRED-MD/QD datasets
│   ├── execution.py            # Model training and evaluation workflow
│   