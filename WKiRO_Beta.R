library(mlr)
library(pROC)
source("functions.R")

DATA_DIR <- "data"
IRIS_DATA_FILE_NAME <- "glass.dataWithNames"
IRIS_CLASS_COLLUMN_NAME <- "class"
CLASSIF_LDA <- "classif.lda"
CLASSIF_SVM <- "classif.svm"
CLASSIF_naiveBayes <- "classif.naiveBayes"
CLASSIF_knn <- "classif.knn"
PREDICT_TYPE <- "prob"

data <- getData(IRIS_DATA_FILE_NAME)

task = makeClassifTask(data = data$trainData, target = "class")


lrnSvn = makeLearner(CLASSIF_SVM, predict.type = PREDICT_TYPE, kernel = "sigmoid")
lrnLda = makeLearner(CLASSIF_LDA, predict.type = PREDICT_TYPE )
lrnKnn = makeLearner(CLASSIF_knn)
lrnNaiveBayes = makeLearner(CLASSIF_LDA, predict.type = PREDICT_TYPE)


modelSvn = train(lrnSvn, task)
modelLda = train(lrnLda, task)
modelNaiveBayes = train(lrnNaiveBayes, task)
modelKnn = train(lrnKnn, task)

wynik = getMAndCi(trainAndPredict(lrnSvn,task,data$testData),data$testData$class)
print(wynik$M)
print(wynik$Ci)
wynik = getMAndCi(trainAndPredict(lrnLda,task,data$testData),data$testData$class)
print(wynik$M)
print(wynik$Ci)
wynik = getMAndCi(trainAndPredict(lrnNaiveBayes,task,data$testData),data$testData$class)
print(wynik$M)
print(wynik$Ci)
wynik = wynik = getMAndCi(trainAndPredict(lrnKnn,task,data$testData),data$testData$class)
print(wynik$M)
print(wynik$Ci)


