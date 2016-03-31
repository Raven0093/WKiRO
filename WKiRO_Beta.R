library(mlr)
library(pROC)
source("functions.R")

DATA_DIR <- "data"

CAR_DATA_FILE_NAME <- "car.dataWithNames"
CAR_NUMERICAL_DATA_FILE_NAME <- "car.numericalDataWithNames"
GLASS_DATA_FILE_NAME <- "glass.dataWithNames"
IRIS_DATA_FILE_NAME <- "iris.dataWithNames"
OPTDIGITS_DATA_FILE_NAME <- "optdigits.dataWithNames"
VOWEL_DATA_FILE_NAME <- "vowel.dataWithName"

IRIS_CLASS_COLLUMN_NAME <- "class"
CLASSIF_LDA <- "classif.lda"
CLASSIF_SVM <- "classif.svm"
CLASSIF_naiveBayes <- "classif.naiveBayes"
CLASSIF_knn <- "classif.knn"
PREDICT_TYPE <- "prob"

data <- getData(IRIS_DATA_FILE_NAME)

task = makeClassifTask(data = data$trainData, target = "class")


knnClassif(task,data$testData)
ldaClassif(task,data$testData)
svmClassif(task,data$testData)
naiveBayesClassif(task,data$testData)


