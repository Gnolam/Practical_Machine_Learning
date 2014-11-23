rm(list=ls())


library(caret)
library(doParallel)
library(dplyr)
library(randomForest)
library(e1071)
library(Hmisc)

# Parameters
set.seed(4394)

fname.test   <- "pml-testing.csv"
fname.train  <- "pml-training.csv"


# Function to be used later

## Loading data with NAs, empty values and #DIV/0! should be treated as NAs
read.dataset <- function(x) { read.csv(x, na.strings = c("NA", "", "#DIV/0!") ) }


## Removing NAs from columns and rows
keep.only.clean.cols  <- function(x) {x[,sapply(x, function(y) !any(is.na(y)))] }




# Load data
setwd("D:/Coursera/ML/Project1")

training.set   <-  read.dataset(fname.train)
test.set       <-  read.dataset(fname.test)


#   Clean data

##  Remove any column containing NA value
test.set     <- keep.only.clean.cols(test.set)
training.set <- keep.only.clean.cols(training.set)


## Remove irrelevant columns
test.set     <- test.set[,-c(1:6)]
training.set <- training.set[,-c(1:6)]


##  Split training data into a training and test subsets
trainingIndex  <- createDataPartition(training.set$classe, p=.7, list=FALSE)
training.train       <- training.set[ trainingIndex,]
training.validation  <- training.set[-trainingIndex,]


#
registerDoParallel()


x <- select(training.train,-classe)
y <- training.train$classe

rf <-
  foreach(
    ntree = rep(150, 4),
    .combine = randomForest::combine,
    .packages = 'randomForest') %dopar% {randomForest(x, y, ntree=ntree)}



predictions1 <- predict(rf, newdata=training.train)
confusionMatrix(predictions1,training.train$classe)


predictions2 <- predict(rf, newdata=training.validation)
confusionMatrix(predictions2,training.validation$classe)

