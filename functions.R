getData <- function(FileName){
  
  importData <- read.csv(paste0(DATA_DIR,"/",FileName),header = TRUE)
  rowNumber = nrow(importData)
  
  train.set = sort(sample(rowNumber, size = round(2/3 * rowNumber)))
  test.set = sort(setdiff(seq_len(rowNumber), train.set))
  
  trainData = importData[train.set,]
  testData = importData[test.set,]
  
  rownames(trainData) <- seq(length=nrow(trainData))
  rownames(testData) <- seq(length=nrow(testData))

  return(list("trainData" = trainData,"testData" = testData))
}

getMAndCi <- function(predictions,target){
  predictionsResponse <- as.numeric(getPredictionResponse(predictions))
  multiclassResult = multiclass.roc(target, predictionsResponse)
  aucResult = auc(multiclassResult)
  ci = ci.auc(target,predictionsResponse)
  return(list("M" = aucResult, "Ci" = ci))
}

trainAndPredict <-function(lrn, task, data){
  model= train(lrn, task)
  predictions = predict(model, newdata = data)
  return(predictions)
}
knnClassif <- function(task,testData){
  print("KNN****************************")
  lrnKnn = makeLearner(CLASSIF_knn, k = 9)
  wynik = wynik = getMAndCi(trainAndPredict(lrnKnn,task,testData),testData$class)
  print(wynik$M)
  print(wynik$Ci)
  print("--------------")
}
ldaClassif <- function(task,testData){
  print("LDA****************************")
  lrnLda = makeLearner(CLASSIF_LDA, predict.type = PREDICT_TYPE)
  wynik = wynik = getMAndCi(trainAndPredict(lrnLda,task,testData),testData$class)
  print(wynik$M)
  print(wynik$Ci)
  print("--------------")
}

svmClassif <- function(task,testData){
  print("SVM****************************")
  lrnSvn = makeLearner(CLASSIF_SVM, predict.type = PREDICT_TYPE, kernel = "sigmoid")
  wynik = wynik = getMAndCi(trainAndPredict(lrnSvn,task,testData),testData$class)
  print(wynik$M)
  print(wynik$Ci)
  print("--------------")
}
naiveBayesClassif <- function(task,testData){
  print("NaiveBayes****************************")
  lrnNaiveBayes = makeLearner(CLASSIF_naiveBayes)
  wynik = wynik = getMAndCi(trainAndPredict(lrnNaiveBayes,task,testData),testData$class)
  print(wynik$M)
  print(wynik$Ci)
  print("--------------")
}


# knnClassif <- function(task,testData){
#   print("KNN****************************")
#   for (i in 1:50){
#     if((i %% 6) == 0){
#       lrnKnn = makeLearner(CLASSIF_knn, k = i)
#       wynik = wynik = getMAndCi(trainAndPredict(lrnKnn,task,testData),testData$class)
#       print (c("knn k = ",i))
#       print(wynik$M)
#       print(wynik$Ci)
#       print("--------------")
#     }
#     
#   }
# }