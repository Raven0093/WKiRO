library(mlr)
library(pROC)
source("functions.R")
for(i in 0:9){
DATA_DIR <- "data"
IRIS_DATA_FILE_NAME <- "optdigits.dataWithNames"
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
}


