import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdmpy 
import statsmodels.api as sm
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
import sys
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from fredapi import Fred
import os
import time

VERBOSE = False  # Set to False to reduce print output