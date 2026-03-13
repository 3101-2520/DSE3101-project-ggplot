#<<<<<<< HEAD
# print('hi')
#=======
### Install packages

install.packages(c("readr", "pracma", "fbi"))

library(stats)
library(readr)
library(pracma)


## Load in Data

dataMD_2026_2 <- read.csv("2026-02-MD.csv")

head(data_2026_2)


# Load the entire FRED-MD dataset (135 monthly variables)
fred_data <- fredmd(file = "../data/2026-01-MD.csv", transform = TRUE)

# Check what you got
class(fred_data)  # Should be "fredmd"
dim(fred_data)    # [Months × 135 variables]
head(fred_data)

# See variable names and transformations
data(fredmd_description)
View(fredmd_description)







