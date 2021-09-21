library(NLoN)
training_set <- fread("04_training_set.csv", encoding="UTF-8")
test_set <- fread("04_test_set.csv", encoding="UTF-8")
validation_set <- fread("full_validation_set.csv", encoding="UTF-8")

print('lines that equal "NA" will lead to an exception and should be removed from datasets')

print('model training')
model <- NLoNModel(training_set$doc, training_set$target)
print('done model training')

print('predict test set')
test_set$nlonPrediction <- NLoNPredict(model, test_set$doc)
fwrite(test_set,"full_test_set_nlon_predicted.csv")

print('predict validation set set')
validation_set$nlonPrediction <- NLoNPredict(model, validation_set$doc)
fwrite(validation_set,"full_validation_set_nlon_predicted.csv")
