---
title: "Practical Machine Learning - Prediction Assignment Writeup"
author: "Alex Skorokhod"
date: "Monday, November 24, 2014"
output: html_document
---

## Background

Using devices such as *Jawbone Up, Nike FuelBand*, and *Fitbit* it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har>. 


## Structure of the code
1. List of libraries to be included in the project
2. Definition of the parameters and functions to be used
3. Loading data
4. Cleaning data
5. Definition of training and validation set
6. Running forest tree + parallel
7. Examination of results


## 1. Libraries used in the project
```{r message=FALSE}
library(caret)
library(doParallel)
library(dplyr)
library(randomForest)
library(e1071)  # required implicitly
library(Hmisc)
```


## 2. Definition of the parameters and functions to be used
First we have to clean all previous calculations and reset seed counter to
ensure the reproduceble nature of the excercise:
```{r message=FALSE}
rm(list=ls())
set.seed(4394)
```

It is more convenient and error-proof to set up the filenames here as parameters:
```{r message=FALSE}
fname.test   <- "pml-testing.csv"
fname.train  <- "pml-training.csv"
```

Function to be used later:
```{r message=FALSE}
## Loading data with NAs, empty values and #DIV/0! should be treated as NAs
read.dataset <- function(x) { read.csv(x, na.strings = c("NA", "", "#DIV/0!") ) }
```

```{r message=FALSE}
## Removing NAs from columns and rows
keep.only.clean.cols  <- function(x) {x[,sapply(x, function(y) !any(is.na(y)))] }
```


## 3. Loading data
Loading data with NAs, empty values and #DIV/0! should be treated as NAs:
```{r message=FALSE}
training.set   <-  read.dataset(fname.train)
test.set       <-  read.dataset(fname.test)
```


## 4. Cleaning data
We remove any columns which contain at least 1 implied *NA* value:
```{r message=FALSE}
##  Remove any column containing "", NA or division by zero value
test.set     <- keep.only.clean.cols(test.set)
training.set <- keep.only.clean.cols(training.set)
```

First 6 columns do not contain information suitable for modelling.
Ideally we have to address them by names.
```{r message=FALSE}
## Remove irrelevant columns
test.set     <- test.set[,-c(1:6)]
training.set <- training.set[,-c(1:6)]
```



## 5. Definition of training and validation set
We split training data into a training and validation samples in the proportion of 70%:30%:
```{r message=FALSE}
##  Split training data into a training and test subsets
trainingIndex  <- createDataPartition(training.set$classe, p=.7, list=FALSE)
training.train       <- training.set[ trainingIndex,]
training.validation  <- training.set[-trainingIndex,]
```


## 6. Running forest tree + parallel
I have googled this part for greater efficiency.

Now lets build 150x4=600 random forests. Thanks to *doParallel* package we can use multiple cores of the CPU:
```{r message=FALSE}
registerDoParallel()

x <- select(training.train,-classe)
y <- training.train$classe

rf <-
  foreach(
    ntree = rep(150, 4),
    .combine = randomForest::combine,
    .packages = 'randomForest') %dopar% {randomForest(x, y, ntree=ntree)}
```


## 7. Examination of results
Now we can estimate model fit:


For training data:
```{r message=FALSE}
prediction.train <- predict(rf, newdata=training.train)
confusionMatrix(prediction.train,training.train$classe)
```


For validation sample:
```{r message=FALSE}
prediction.validation <- predict(rf, newdata=training.validation)
confusionMatrix(prediction.validation,training.validation$classe)
```

## Colnclusion
There is no surpise that random forest resulted into 100% accuracy for the training test (Kappa=1.0)

What is more important is accuracy of 99.66% for the validation sample (Kappa=0.9957).

With such results I am pretty confident to proceeding to a submission phase.
