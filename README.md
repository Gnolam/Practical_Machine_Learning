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

```r
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

```r
rm(list=ls())
set.seed(4394)
```

It is more convenient and error-proof to set up the filenames here as parameters:

```r
fname.test   <- "pml-testing.csv"
fname.train  <- "pml-training.csv"
```

Function to be used later:

```r
## Loading data with NAs, empty values and #DIV/0! should be treated as NAs
read.dataset <- function(x) { read.csv(x, na.strings = c("NA", "", "#DIV/0!") ) }
```


```r
## Removing NAs from columns and rows
keep.only.clean.cols  <- function(x) {x[,sapply(x, function(y) !any(is.na(y)))] }
```


## 3. Loading data
Loading data with NAs, empty values and #DIV/0! should be treated as NAs:

```r
training.set   <-  read.dataset(fname.train)
test.set       <-  read.dataset(fname.test)
```


## 4. Cleaning data
We remove any columns which contain at least 1 implied *NA* value:

```r
##  Remove any column containing "", NA or division by zero value
test.set     <- keep.only.clean.cols(test.set)
training.set <- keep.only.clean.cols(training.set)
```

First 6 columns do not contain information suitable for modelling.
Ideally we have to address them by names.

```r
## Remove irrelevant columns
test.set     <- test.set[,-c(1:6)]
training.set <- training.set[,-c(1:6)]
```



## 5. Definition of training and validation set
We split training data into a training and validation samples in the proportion of 70%:30%:

```r
##  Split training data into a training and test subsets
trainingIndex  <- createDataPartition(training.set$classe, p=.7, list=FALSE)
training.train       <- training.set[ trainingIndex,]
training.validation  <- training.set[-trainingIndex,]
```


## 6. Running forest tree + parallel
I have googled this part for greater efficiency.

Now lets build 150x4=600 random forests. Thanks to *doParallel* package we can use multiple cores of the CPU:

```r
registerDoParallel()

x <- select(training.train,-classe)
y <- training.train$classe

rf <-
  foreach(
    ntree = rep(150, 4),
    .combine = randomForest::combine,
    .packages = 'randomForest') %dopar% {randomForest(x, y, ntree=ntree)}
```

```
## Warning: closing unused connection 5 (<-WSSYD3LP4850.ais.local:11462)
```

```
## Warning: closing unused connection 4 (<-WSSYD3LP4850.ais.local:11462)
```

```
## Warning: closing unused connection 3 (<-WSSYD3LP4850.ais.local:11462)
```


## 7. Examination of results
Now we can estimate model fit:


For training data:

```r
prediction.train <- predict(rf, newdata=training.train)
confusionMatrix(prediction.train,training.train$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```


For validation sample:

```r
prediction.validation <- predict(rf, newdata=training.validation)
confusionMatrix(prediction.validation,training.validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    3    0    0    0
##          B    0 1136    7    0    0
##          C    0    0 1019    6    0
##          D    0    0    0  958    4
##          E    0    0    0    0 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9966          
##                  95% CI : (0.9948, 0.9979)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9957          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9932   0.9938   0.9963
## Specificity            0.9993   0.9985   0.9988   0.9992   1.0000
## Pos Pred Value         0.9982   0.9939   0.9941   0.9958   1.0000
## Neg Pred Value         1.0000   0.9994   0.9986   0.9988   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1930   0.1732   0.1628   0.1832
## Detection Prevalence   0.2850   0.1942   0.1742   0.1635   0.1832
## Balanced Accuracy      0.9996   0.9979   0.9960   0.9965   0.9982
```

## Colnclusion
There is no surpise that random forest resulted into 100% accuracy for the training test (Kappa=1.0)

What is more important is accuracy of 99.66% for the validation sample (Kappa=0.9957).

With such results I am pretty confident to proceeding to a submission phase.
