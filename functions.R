#Pobieranie danych oraz dzielenie na dane testowe oraz treningowe
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

#Generowanie obiektów z funkcji multiclass.roc oraz ci.auc
getMAndCi <- function(predictions,target){
  predictionsResponse <- as.numeric(getPredictionResponse(predictions))
  multiclassResult = multiclass.roc(target, predictionsResponse)
  aucResult = auc(multiclassResult)
  ci = ci.auc(target,predictionsResponse)
  return(list("M" = aucResult, "Ci" = ci))
}

#trenowanie oraz przewidywanie
trainAndPredict <-function(lrn, task, data){
  model= train(lrn, task)
  predictions = predict(model, newdata = data)
  return(predictions)
}
#generowanie wyników dla klasyfikatora KNN
knnClassif <- function(task,testData){
  print("KNN")
  lrnKnn = makeLearner(CLASSIF_knn, k = 9)
  result = getMAndCi(trainAndPredict(lrnKnn,task,testData),testData$class)
  print(result$M)
  print(result$Ci)
}
#generowanie wyników dla klasyfikatora LDA
ldaClassif <- function(task,testData){
  print("LDA")
  lrnLda = makeLearner(CLASSIF_LDA, predict.type = PREDICT_TYPE)
  result = getMAndCi(trainAndPredict(lrnLda,task,testData),testData$class)
  print(result$M)
  print(result$Ci)
}
#generowanie wyników dla klasyfikatora SVM
svmClassif <- function(task,testData){
  print("SVM")
  lrnSvn = makeLearner(CLASSIF_SVM, predict.type = PREDICT_TYPE, kernel = "sigmoid")
  result = getMAndCi(trainAndPredict(lrnSvn,task,testData),testData$class)
  print(result$M)
  print(result$Ci)
}
#generowanie wyników dla klasyfikatora naiveBayes
naiveBayesClassif <- function(task,testData){
  print("NaiveBayes")
  lrnNaiveBayes = makeLearner(CLASSIF_naiveBayes)
  result = getMAndCi(trainAndPredict(lrnNaiveBayes,task,testData),testData$class)
  print(result$M)
  print(result$Ci)
}

