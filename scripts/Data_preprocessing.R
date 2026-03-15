### Install packages

## install.packages(c("readr", "pracma", "fbi"))

library(stats)
library(readr)
library(pracma)
library(fbi)

## Load in Data

data_2026_1 <- read.csv("../data/2026-01-MD.csv")

head(data_2026_1)


# Load the entire FRED-MD dataset (135 monthly variables)
fred_data <- fredmd(file = "../data/2026-01-MD.csv", transform = TRUE)

# Check what you got
class(fred_data)  # Should be "fredmd"
dim(fred_data)    # [Months × 135 variables]
head(fred_data)

# See variable names and transformations
data(fredmd_description)
View(fredmd_description)






