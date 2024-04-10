----
# Predicting Product Subscription for a Portuguese Bank
---


### Project Overview
This project aims to develop a reliable predictive model to help a Portuguese bank optimize its resources by targeting individuals more likely to subscribe to a new product line it intends to introduce.



## Table of Contents
- [Introduction](#introduction)
- [Tools Used](#tools-used)
- [Data Source](#data-source)
- [Data Description](#data-description)
- [Data Cleaning & Preparation](#data-cleaning-&-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
    - [Logistics Regression Model (LRM)](#logistics-regression-model(lrm))
- [Findings & Results](#findings-&-results)
- [Recommendations](#recommendations)
- [Limitations](#limitations)



### Introduction
* The task is to develop a model that accurately predicts whether an individual will likely subscribe to a new product.
* Algorithms/Methods: Logistics Regression, and Random Forest.
* The forecasts will be assessed using measures of forecast evaluation such as accuracy, precision, recall, and F1-score.



### Tools Used
- Microsoft Excel
- R Programming


### Data Source
* The extracted dataset contains 17 variables of anonymized customer data, categorized into; Customer Data (1-8), Contact Data (9-12), Contact Summary Data (13-16), and Result (17). The full dataset is described and analyzed in S. Moro, P. Cortez, and P. Rita. A Data|Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems (2014), doi:10.1016/j.dss.2014.03.001


 
### Data Description
| S/N | Field Name | Description | Data Type |
|-----|----------|-----------|---------|
| 1. | age | respondent's age | numeric |
| 2. | job | type of job | categorical |
| 3. | marital | marital status | categorical |
| 4. | education | level of education | categorical |
| 5. | default | any credit defaults?|categorical |
| 6. | balance | balance on the current account | numeric |
| 7. | housing | has housing loan? | categorical |
| 8. | loan | has personal loan? | categorical |
| 9. | contact | contact communication type | categorical |
| 10. | day | contact day of week | categorical |
| 11. | month | contact month of the year | categorical |
| 12. | duration | contact duration, in seconds | numeric |
| 13. | campaign | number of contacts performed during this campaign and for this respondent | numeric, includes last contact |
| 14. | pdays | no of days since the client was last contacted from a previous campaign | numeric, 1 means respondent not previously contacted |
| 15. | previous | number of contacts performed before this campaign and for this client | numeric |
| 16. | poutcome | outcome of the previous marketing campaign | categorical |
| 17. | y (target variable) | has the client subscribed to the new product? | Binary (0 and 1) |



### Data Cleaning & Preparation


Set and confirm the working directory
```{r}
setwd("C:/Users/ariso/Documents/Pet Projects/Predictive Analytics_R/Predicting Cust Subscription");

getwd() # to confirm directory in use;
```


Import the dataset
```{r}
data <- read_xlsx("data_pbank.xlsx")
```


Load necessary libraries
```{r}
library (readxl)
library (readr)
library(randomForest)
library(Metrics)
library(lattice)
library(caret)
library(rpart)
library(ModelMetrics)
library(dplyr)
library(forecast)
library(car)
library(MASS) 
library(e1071) 
library(ggplot2) 
library(lattice) 
library(tidyverse) 
library(broom) 
library(purrr);
```


Evaluate the variables and their current data types
```{r}
str(data)
```
![image](https://github.com/MrArisoyin/Damilola-s-Analytics/assets/65539376/92407a13-fef4-4732-9d1e-dea5a7d3a863)



View information on the outcome variable
```{r}
table(data$y);
# 0 = unsubscribed customers
# 1 = subscribed customers
```
![image](https://github.com/MrArisoyin/Damilola-s-Analytics/assets/65539376/30a0ddb9-6791-4fe5-ac4b-9b42cfbfb861)



**Pre-processing and cleaning the data**

Checking for Missing/Null Values
```{r}
colSums(is.na(data)) # Result is 0; no missing values
```
![image](https://github.com/MrArisoyin/Damilola-s-Analytics/assets/65539376/2406f47a-dca3-48dc-99af-0d05368e6c05)


#### Detecting Outliers
**Numerical Variables**
```{r}
# logarithmic transformation of the numerical variables to even the scale
data$log_age <- log(data$age)
data$log_balance <- log(data$balance)
data$log_day <- log(data$day)
data$log_duration <- log(data$duration)
data$log_pdays <- log(data$pdays)

boxplot(data[, c("log_age", "log_balance", "log_day", "log_duration", "log_pdays")],
        col = "red",
        main = "Detecting Outliers",
        xlab = "Variables", ylab = "Values",
        names = c("age", "bal", "day", "duratn", "pdays"))
```
![image](https://github.com/MrArisoyin/Damilola-s-Analytics/assets/65539376/75a4e01f-46f2-416e-909d-84d28e3cae36)

Upon scrutiny of the above box plot, it is observed that the...*


**Categorical and other non-numerical variables**
```{r}
df <- data[c("job", "marital", "education", "default", "housing", "loan", "contact", "month", "campaign", "previous", "poutcome", "y")]

table(df)

table(data$job) 
table(data$marital)
table(data$education)
table(data$default)
table(data$housing)
table(data$loan)
table(data$contact)
table(data$month)
table(data$campaign)

table(data$previous)
table(data$poutcome)
table(data$y)
```
![image](https://github.com/MrArisoyin/Damilola-s-Analytics/assets/65539376/1d4c9146-4af2-4f21-9810-de52b1226b94)



### Exploratory Data Analysis
* Question 1
* Question 2
* Question 3



### Model Building

**Data Type Conversion**
Convert the categorical data variables in the data set to factors;
```{r}
data$job <- as.factor(data$job)
data$marital <- as.factor(data$marital)
data$education <- as.factor(data$education)
data$default <- as.factor(data$default)
data$housing <- as.factor(data$housing)
data$loan <- as.factor(data$loan)
data$contact <- as.factor(data$contact)
data$month <- as.factor(data$month)
data$poutcome <- as.factor(data$poutcome)
data$y <- as.factor(data$y)
```


**Split the data into train and test datasets**

```{r}
set.seed(99) #Football reference, we shouldn't forget False 9s

# create an index for splitting the data into train and test data  sets
index <- createDataPartition(data$y, p = 0.7, list = FALSE, times = 1);

# split the data into train and test sets based on the index
train <- data[index, ]
test <- data[-index, ];
```

**Logistics Regression Model (LRM)**
