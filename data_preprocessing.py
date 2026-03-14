import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MD = pd.read_csv("../data/2026-02-MD.csv")
QD = pd.read_csv("../data/2026-02-QD.csv")

print(MD.shape)
print(MD.columns)
print(MD.head())
print(MD.info())

print(QD.shape)
print(QD.columns)
print(QD.head())
print(QD.info())


# transformation code extraction 
tcodes = MD.iloc[0]
MD = MD.iloc[1:]

MD['sasdate'] = pd.to_datetime(MD['sasdate'], format='%m/%d/%Y')
MD = MD.set_index('sasdate')

def transform_series(series, code):
    if code == 1:
        return series

    elif code == 2:
        return series.diff()

    elif code == 3:
        return series.diff().diff()

    elif code == 4:
        return np.log(series)

    elif code == 5:
        return np.log(series).diff()

    elif code == 6:
        return np.log(series).diff().diff()
    
    else:
        return series
    

### apply transformations to all pred
# 1. Create a list to hold each transformed series
transformed_list = []

# 2. Run the loop to transform each column
for col in MD.columns:
    code = int(tcodes[col])
    # Apply transformation and keep the series name
    s = transform_series(MD[col].astype(float), code)
    s.name = col
    transformed_list.append(s)

# 3. "Stitch" them all together at once (This prevents fragmentation)
MD_trans = pd.concat(transformed_list, axis=1)

# 4. Final cleaning (from Slide 46)
MD_trans.dropna(inplace=True)

print("EDA Step 1 Complete: Data is stationary and de-fragmented.")


# Compare "Real Personal Income" (typically TCODE 5)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(MD['RPI'])
plt.title("Original RPI (Non-Stationary)")

plt.subplot(1, 2, 2)
plt.plot(MD_trans['RPI'])
plt.title("Transformed RPI (Stationary)")
plt.show()



