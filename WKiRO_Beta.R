library(mlr)
library(pROC)

DATA_DIR <- "data"
IRIS_DATA_FILE_NAME <- "iris.data"
IRIS_CLASS_COLLUMN_NAME <- "Species"
CLASSIF <- "classif.lda"
PREDICT_TYPE <- "prob"


importData <- read.csv(paste0(DATA_DIR,"/",IRIS_DATA_FILE_NAME),header = TRUE)
rowNumber = nrow(importData)

train.set = sort(sample(rowNumber, size = round(2/3 * rowNumber)))
test.set = sort(setdiff(seq_len(rowNumber), train.set))

trainData = importData[train.set,]
testData = importData[test.set,]

rownames(trainData) <- seq(length=nrow(trainData))
rownames(testData) <- seq(length=nrow(testData))

task = makeClassifTask(data = trainData, target = IRIS_CLASS_COLLUMN_NAME)
lrn = makeLearner(CLASSIF, predict.type = PREDICT_TYPE)



model = train(lrn, task)

predictions = predict(model, newdata = testData)

#multiclassResult = multiclass.roc(testData$Species, predictions)

#predicted.class <- apply(mnm.predict.test.probs, 1, which.max)

wynik = multiclass.roc(testData$Species,as.numeric(predictions$data$response))

#multiclass.roc(ProxFiltered$response_variable, apply(preds2, 1, function(row) which.max(row)))


