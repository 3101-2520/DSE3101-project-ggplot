# DSE3101 gg(plot) 

## Research Topic

Nowcasting the economy

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

## Project Structure

TODO

## Local Setup

1. **Create a Python virtual environment** (recommended):

   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **FRED API Setup**
   1. Create a FRED account and request an API key at https://fred.stlouisfed.org/docs/api/fred/v2/api_key.html
   2. Set the API key as an enviroment variable
      - On macOS/Linux:
     ```bash
     export FRED_API_KEY = "your_api_key_here"
     ```
   - On Windows:
     ```bash
     $env:FRED_API_KEY="your_api_key_here"
     ```
   3. Check that your API key has been set
      - On macOS/Linux:
     ```bash
     echo $FRED_API_KEY
     ```
   - On Windows:
     ```bash
     echo $env:FRED_API_KEY
     ```
## Running files

The project has 2 main workflows:

1. **Backend Pipeline**
   ```bash
   python -m src.execution
   ```

2. **Live Nowcasting Pipeline**
   1. Fetch latest FRED data
      ```bash
      python -m src.api_preprocessing
      ```
   2. Run live nowcast
      ```bash
      python -m src.live_nowcast
      ```

## Contributed By

1. Bryce Tan Jing Kai (A0272764W)
2. Chin Chen Shao Javier (A0272898E)
3. Gan Zhi Yu Charlene (A0282072J)
4. Melanie Tan Yong En (A0277113J)
5. Owen Lim Wen Xuan (A0272606E) 
6. Shannon Kwok En Yi (A0281617B)
7. Tan Sze Ping (A0286558J)
8. Vanisha Muthu (A0282716Y)