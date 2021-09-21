library(NLoN)
model <- NLoNModel(nlon.data$text, nlon.data$rater2)

res <- fread("manual_validation_set_reviewer_2.csv", encoding="UTF-8")

res$nlonPrediction <- NLoNPredict(model, res$report)

fwrite(res,"pretrained_nlon_predict_validation_set.csv")