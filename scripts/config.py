
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdmpy 
import statsmodels.api as sm
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import StandardScaler

# You can also define global paths here so you don't repeat them
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"