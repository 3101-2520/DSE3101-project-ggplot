import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdmpy as hdm

MD = pd.read_csv("../data/2026-02-MD.csv")
QD = pd.read_csv("../data/2026-02-QD.csv")

#print(MD.shape)
#print(MD.columns)
#print(MD.head())
#print(MD.info())

#print(QD.shape)
#print(QD.columns)
#print(QD.head())
#print(QD.info())


# transformation code extraction 
tcodes_md = MD.iloc[0]
MD = MD.iloc[1:]

MD['sasdate'] = pd.to_datetime(MD['sasdate'], format='%m/%d/%Y')
MD = MD.set_index('sasdate')


tcodes_qd =  QD.iloc[0]
QD = QD.iloc[1:].copy()
QD = QD[QD['sasdate'].str.lower() != 'transform']

QD['sasdate'] = pd.to_datetime(QD['sasdate'], format = '%m/%d/%Y')
QD = QD.set_index('sasdate')


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
    

### apply transformations to all MD pred
def tcode_trans(DF, tcodes):
# 1. Create a list to hold each transformed series
    transformed_list = []
    # 2. Run the loop to transform each column
    for col in DF.columns:
        code = int(tcodes[col])
        # Apply transformation and keep the series name
        s = transform_series(DF[col].astype(float), code)
        s.name = col
        transformed_list.append(s)
    # 3. "Stitch" them all together at once (This prevents fragmentation)
    DF_trans = pd.concat(transformed_list, axis=1)
    # 4. Final cleaning (from Slide 46)
    DF_trans.dropna(inplace=True)

    return DF_trans

MD_trans = tcode_trans(MD, tcodes_md)
QD_trans = tcode_trans(QD, tcodes_qd)

print("EDA Step 1 Complete: Data is stationary and de-fragmented.")


# Compare "Real Personal Income" (typically TCODE 5)
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(MD['RPI'])
# plt.title("Original RPI (Non-Stationary)")

# plt.subplot(1, 2, 2)
# plt.plot(MD_trans['RPI'])
# plt.title("Transformed RPI (Stationary)")
# plt.show() 

### apply transformations to all QD pred
# transformed_qd_list = []

# for col in QD.columns:
#     code = int(tcodes_qd[col])
    
#     s = transform_series(QD[col].astype(float), code)
#     s.name = col
    
#     transformed_qd_list.append(s)

# QD_trans = pd.concat(transformed_qd_list, axis=1)
# QD_trans.dropna(inplace=True)

# GDP = QD_trans['GDPC1']