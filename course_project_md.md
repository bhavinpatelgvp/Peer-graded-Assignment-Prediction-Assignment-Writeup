course project
================

``` r
#Loading Packages
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(knitr)
library(rpart)
library(rpart.plot)
library(rattle)
```

    ## Rattle: A free graphical interface for data science with R.
    ## Versión 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
    ## Escriba 'rattle()' para agitar, sacudir y  rotar sus datos.

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:rattle':
    ## 
    ##     importance

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
set.seed(123)


######################################################
# Getting and Cleaning Data
######################################################

# Preparing for download
Trainurl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
Testurl  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download the datasets
Training <- read.csv(url(Trainurl))
Testing  <- read.csv(url(Testurl))

# create a partition with the Training dataset 
inTrain  <- createDataPartition(Training$classe, p=0.7, list=FALSE)
Train <- Training[inTrain, ]
Test  <- Training[-inTrain, ]

# Cleaning variables with variables that have near zero values
NZV <- nearZeroVar(Train)
Train <- Train[, -NZV]
Test  <- Test[, -NZV]

# Removing the nomenclature columns
Train <- Train[, -(1:5)]
Test  <- Test[, -(1:5)]

# Removing variables with too many NA values, 90% NA or more
ToomanyNA    <- sapply(Train, function(x) mean(is.na(x))) > 0.90
Train <- Train[, ToomanyNA==FALSE]
Test  <- Test[, ToomanyNA==FALSE]



######################################################
# Prediction with Random Forests
######################################################

# Model Fit
controlrf <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitrf <- train(classe ~ ., data=Train, method="rf",
                          trControl=controlrf)
modFitrf$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.23%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3904    1    0    0    1 0.0005120328
    ## B    7 2649    2    0    0 0.0033860045
    ## C    0    5 2391    0    0 0.0020868114
    ## D    0    0   11 2241    0 0.0048845471
    ## E    0    0    0    4 2521 0.0015841584

``` r
# Prediction on Test
predictrf <- predict(modFitrf, newdata=Test)
confMatrf <- confusionMatrix(predictrf, Test$classe)
confMatrf
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    3    0    0    0
    ##          B    0 1135    4    0    0
    ##          C    0    0 1022    4    0
    ##          D    0    1    0  959    0
    ##          E    0    0    0    1 1082
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9978          
    ##                  95% CI : (0.9962, 0.9988)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9972          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9965   0.9961   0.9948   1.0000
    ## Specificity            0.9993   0.9992   0.9992   0.9998   0.9998
    ## Pos Pred Value         0.9982   0.9965   0.9961   0.9990   0.9991
    ## Neg Pred Value         1.0000   0.9992   0.9992   0.9990   1.0000
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1929   0.1737   0.1630   0.1839
    ## Detection Prevalence   0.2850   0.1935   0.1743   0.1631   0.1840
    ## Balanced Accuracy      0.9996   0.9978   0.9976   0.9973   0.9999

``` r
# Plot
plot(confMatrf$table, col = confMatrf$byClass, 
     main = paste("Random Forest Accuracy =",
                  round(confMatrf$overall['Accuracy'], 4)))
```

![](course_project_md_files/figure-markdown_github/cars-1.png)

``` r
######################################################
# Prediction with Decision Trees
######################################################

# Model Fit
modFitdt <- rpart(classe ~ ., data=Train, method="class")
fancyRpartPlot(modFitdt)
```

    ## Warning: labs do not fit even at cex 0.15, there may be some overplotting

![](course_project_md_files/figure-markdown_github/cars-2.png)

``` r
# Prediction on Test
predictdf <- predict(modFitdt, newdata=Test, type="class")
confMatdf <- confusionMatrix(predictdf, Test$classe)
confMatdf
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1496  114    3   18    7
    ##          B   74  843   55   71   44
    ##          C    0   57  830   37    3
    ##          D   84   51  124  778   68
    ##          E   20   74   14   60  960
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.8338          
    ##                  95% CI : (0.8241, 0.8432)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.7901          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8937   0.7401   0.8090   0.8071   0.8872
    ## Specificity            0.9663   0.9486   0.9800   0.9336   0.9650
    ## Pos Pred Value         0.9133   0.7755   0.8954   0.7041   0.8511
    ## Neg Pred Value         0.9581   0.9383   0.9605   0.9611   0.9744
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2542   0.1432   0.1410   0.1322   0.1631
    ## Detection Prevalence   0.2783   0.1847   0.1575   0.1878   0.1917
    ## Balanced Accuracy      0.9300   0.8444   0.8945   0.8703   0.9261

``` r
# Plot
plot(confMatdf$table, col = confMatdf$byClass, 
     main = paste("Decision Tree Accuracy =",
                  round(confMatdf$overall['Accuracy'], 4)))
```

![](course_project_md_files/figure-markdown_github/cars-3.png)

``` r
######################################################
# Prediction with Generalized Boosted Regression
######################################################

# Model Fit
controlgbm <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitgbm <- train(classe ~ ., data=Train, method = "gbm",
                    trControl = controlgbm, verbose = FALSE)
modFitgbm$finalModel
```

    ## A gradient boosted model with multinomial loss function.
    ## 150 iterations were performed.
    ## There were 53 predictors of which 40 had non-zero influence.

``` r
# Prediction on Test
predictgbm <- predict(modFitgbm, newdata=Test)
confMatgbm <- confusionMatrix(predictgbm, Test$classe)
confMatgbm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673   13    0    0    1
    ##          B    1 1117   13    3    1
    ##          C    0    9 1010   18    4
    ##          D    0    0    3  943    8
    ##          E    0    0    0    0 1068
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9874          
    ##                  95% CI : (0.9842, 0.9901)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9841          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9807   0.9844   0.9782   0.9871
    ## Specificity            0.9967   0.9962   0.9936   0.9978   1.0000
    ## Pos Pred Value         0.9917   0.9841   0.9702   0.9885   1.0000
    ## Neg Pred Value         0.9998   0.9954   0.9967   0.9957   0.9971
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1898   0.1716   0.1602   0.1815
    ## Detection Prevalence   0.2867   0.1929   0.1769   0.1621   0.1815
    ## Balanced Accuracy      0.9980   0.9884   0.9890   0.9880   0.9935

``` r
# Plot
plot(confMatgbm$table, col = confMatgbm$byClass, 
     main = paste("Generalized Boosted Regression Accuracy =", round(confMatgbm$overall['Accuracy'], 4)))
```

![](course_project_md_files/figure-markdown_github/cars-4.png)

``` r
######################################################
# Applying most accurate model (Random Forest)
######################################################

predictionrf<- predict(modFitrf, newdata=Testing)
predictionrf
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
