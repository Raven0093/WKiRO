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
  multiclassResult = multiclass.roc(as.numeric(target), predictionsResponse)
  aucResult = auc(multiclassResult)
  ci = ci(predictionsResponse, as.numeric(target))
  return(list("M" = aucResult, "Ci" = ci))
}

trainAndPredict <-function(lrn, task, data){
  model= train(lrn, task)
  predictions = predict(model, newdata = data)
  return(predictions)
}