#----------------Final Project: Predicting Bankruptcy----------------

# load data into R
bkData <- read.csv("/Users/tianabryant/Downloads/Bankruptcy.csv")
bk.df <- data.frame(bkData) # convert to data frame

# review data to determine how to proceed
str(bk.df)

library(tidyverse)
library(corrplot)
library(forecast)
library(caret)
library(e1071)
library(FNN)
library(rpart)
library(rpart.plot)

# to determine the which groups convey the same thing
bk.SelectColumns <- select(bk.df, 2, 4:27) #remove ID and Year variables
bk.cor <- cor(bk.SelectColumns)
corrplot(bk.cor, method = "color", addCoef.col="grey", number.cex=0.3)

# R9
# create plots for the positive correlations
ggplot(data=bk.df, aes(x=D, y=R9, color=D))+ 
  geom_point()+
  geom_smooth(method = "lm", color="red")

#R17
ggplot(data=bk.df, aes(x=D, y=R17, color=D))+ 
  geom_point()+
  geom_smooth(method = "lm", color="red")

#R23
ggplot(data=bk.df, aes(x=D, y=R23, color=D))+ 
  geom_point()+
  geom_smooth(method = "lm", color="red")

# box and whisker plots

# R9
ggplot(aes(x=D, y=R9, fill=D), data = bk.df) +
  geom_jitter(alpha=.25) +
  geom_boxplot(alpha=.25, aes(group=D))+
  labs(x= "Bankrupt vs. Not Bankrupt", y= "R9")+
  ggtitle("R9 Boxplot (CURASS/CURDEBT)")

# R17
ggplot(aes(x=D, y=R17, fill=D), data = bk.df) +
  geom_jitter(alpha=.25) +
  geom_boxplot(alpha=.25, aes(group=D))+
  labs(x= "Bankrupt vs. Not Bankrupt", y= "R17")+
  ggtitle("R17 Boxplot (INCDEP/ASSETS)")

# R23
ggplot(aes(x=D, y=R23, fill=D), data = bk.df) +
  geom_jitter(alpha=.25) +
  geom_boxplot(alpha=.25, aes(group=D))+
  labs(x= "Bankrupt vs. Not Bankrupt", y= "R23")+
  ggtitle("R23 Boxplot (WCFO/ASSETS)")

#----------PCA Analysis-----------

# normalizing the data
PCA.bk.df <- prcomp(na.omit(bk.df[,-c(1,3)]), scale. = T)

# examine results, focus on proportion of variance
summary(PCA.bk.df)$importance[2,]

# create functions for plots
PCA.var <- PCA.bk.df$sdev^2
PCA.var.percent <- round(PCA.var/sum(PCA.var)*100,1)
PCA.data <- data.frame(Sample=rownames(PCA.bk.df$x),
                       X=PCA.bk.df$x[,1],
                       Y=PCA.bk.df$x[,2])

# visualize the principal components
barplot(PCA.var.percent, 
        main = "PCA Bar Plot", 
        xlab="Principal Component", 
        ylab="Percent Variation",
        col="lightblue")

# visualization
ggplot(data=PCA.data, aes(x=X, y=Y, label=Sample, color=X, fill=X)) +
  geom_point() +
  xlab(paste("PC1 - ", PCA.var.percent[1], "%", sep="")) +
  ylab(paste("PC2 - ", PCA.var.percent[2], "%", sep="")) +
  ggtitle("Bankruptcy PCA Plot")

#----------Linear Regression -----------

# fit linear regression model
bk.reg <- lm(D~R1+R2+R3+R4+R5+R6+R7+R8+R9+R10+R11+R12+R13+R14+
               R15+R16+R17+R18+R19+R20+R21+R22+R23+R24, data=bk.df)

# confirm results
summary(bk.reg)

#----------Stepwise Regression -----------

# partition the data into training (60%) and validation (40%)
sr.train.index <- sample(c(1:dim(bk.df)[1]), dim(bk.df)[1]*0.6)  
sr.train.df <- bk.df[sr.train.index,]
sr.valid.df <- bk.df[-sr.train.index,]

# use step() to run backward regression.
bk.lm.step.back <- step(bk.reg, direction = "backward")
summary(bk.lm.step.back) # Which variables were dropped?
bk.lm.step.back.pred <- predict(bk.lm.step.back, sr.valid.df)
accuracy(bk.lm.step.back.pred, sr.valid.df$D)

# use step() to run forward regression.
bk.lm.null <- lm(D~1, data = sr.train.df)
bk.lm.step.forward <- step(bk.lm.null, 
                           scope=list(lower=bk.lm.null, upper=bk.reg), 
                           direction = "forward")
summary(bk.lm.step.forward) # Which variables were added?
bk.lm.step.forward.pred <- predict(bk.lm.step.forward, sr.valid.df)
accuracy(bk.lm.step.forward.pred, sr.valid.df$D)

# use step() to run both
bk.lm.step <- step(bk.reg, direction = "both")
summary(bk.lm.step) # Which variables were dropped/added?
bk.lm.step.pred <- predict(bk.lm.step, sr.valid.df)
accuracy(bk.lm.step.pred, sr.valid.df$D)

# data frame creation to compare results 
comparison <- data.frame(
  Backward=c(accuracy(bk.lm.step.back.pred, sr.valid.df$D)),
  Forward= c(accuracy(bk.lm.step.forward.pred, sr.valid.df$D)),
  Both=c(accuracy(bk.lm.step.pred, sr.valid.df$D))
)
rownames(comparison) <-c("ME", "RMSE", "MAE", "MPE", "MAPE")
comparison

# confusion matrix on validation data
SWR.CM <- confusionMatrix(factor(ifelse(bk.lm.step.pred > 0.5, 1, 0)), 
                          factor(sr.valid.df$D))

#----------KNN-----------

# remove columns not needed
clean.bk.df <-bk.df[,-c(1,3)] #less the columns not needed

# partition the data into training (60%) and validation (40%)
knn.train.index <- sample(c(1:dim(clean.bk.df)[1]), dim(clean.bk.df)[1]*0.6)  
knn.train.df <- clean.bk.df[knn.train.index,]
knn.valid.df <- clean.bk.df[-knn.train.index,]

# use preProcess() from the caret package to normalize
norm.values <- preProcess(knn.train.df, method=c("center", "scale"))
train.norm.df <- predict(norm.values, knn.train.df)
valid.norm.df <- predict(norm.values, knn.valid.df)
bk.norm.df<- predict(norm.values, bk.df)

train.nn <- knn(train = train.norm.df, test = valid.norm.df,
                cl = train.norm.df[, "D"], k = 1)
row.names(knn.train.df)[attr(train.nn, "nn.index")]

# initialize a data frame with two columns: k, and accuracy.
accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))

# compute knn for different k on validation.
for(i in 1:14){
  knn.pred <- knn(train.norm.df, valid.norm.df,
                  cl = train.norm.df[, "D"], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, 
                                       factor(valid.norm.df[, "D"]))$overall[1]
}
accuracy.df
plot(accuracy.df)

# Show the confusion matrix for the validation data from using the best k.
knn.pred <- knn(train.norm.df, valid.norm.df,
                cl = train.norm.df[, "D"], k = 5)
KNN.CM <- confusionMatrix(knn.pred, factor(valid.norm.df[, "D"]))
KNN.CM
#----------Classification Tree-----------

# use rpart() to run a classification tree.
# using the predictors from the stepwise regression analysis
class.tree <- rpart(D ~ R2+R3+R9+R11+R12+R13+R14+R15+R16+R17+R20+R22+R24, 
                    data = bk.df,control = rpart.control(maxdepth = 2), 
                    method = "class")

# count number of leaves
length(class.tree$frame$var[class.tree$frame$var == "<leaf>"])

# plot tree
prp(class.tree, type = 1, extra = 1, split.font = 1, varlen = -10)  

# partition
set.seed(1)  
ct.train.index <- sample(c(1:dim(bk.df)[1]), dim(bk.df)[1]*0.6)  
ct.train.df <- bk.df[ct.train.index, ]
ct.valid.df <- bk.df[-ct.train.index, ]

# classification tree
default.ct <- rpart(D ~ R2+R3+R9+R11+R12+R13+R14+R15+R16+R17+R20+R22+R24, 
                    data = ct.train.df, method = "class")
# plot tree
prp(default.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,)

deeper.ct <- rpart(D ~ R2+R3+R9+R11+R12+R13+R14+R15+R16+
                     R17+R20+R22+R24, data = ct.train.df, 
                   method = "class", cp = 0, minsplit = 1)

# count number of leaves
length(deeper.ct$frame$var[deeper.ct$frame$var == "<leaf>"])

# plot tree
prp(deeper.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(deeper.ct$frame$var == "<leaf>", 'gray', 'white'))  

# classify records in the training data.
# set argument type = "class" in predict() to generate predicted class membership.
default.ct.point.pred.train <- predict(default.ct,ct.train.df,type = "class")

# generate confusion matrix for training data
confusionMatrix(default.ct.point.pred.train, as.factor(ct.train.df$D))

### repeat the code for the validation set, and the deeper tree
# classification tree
default.ct2 <- rpart(D ~ R2+R3+R9+R11+R12+R13+R14+R15+R16+R17+R20+R22+R24, 
                     data = ct.valid.df, method = "class")
# plot tree
prp(default.ct2, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,)

deeper.ct2 <- rpart(D ~ R2+R3+R9+R11+R12+R13+R14+R15+R16+
                      R17+R20+R22+R24, data = ct.valid.df, 
                    method = "class", cp = 0, minsplit = 1)

# count number of leaves
length(deeper.ct2$frame$var[deeper.ct$frame$var == "<leaf>"])

# plot tree
prp(deeper.ct2, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(deeper.ct2$frame$var == "<leaf>", 'gray', 'white'))  

# classify records in the validation data.
# set argument type = "class" in predict() to generate predicted class membership.
default.ct.point.pred.valid <- predict(default.ct2,ct.valid.df,type = "class")

# generate confusion matrix for validation data
CT.CM <- confusionMatrix(default.ct.point.pred.valid, as.factor(ct.valid.df$D))

# cross-validation procedure
cv.ct <- rpart(D~R2+R3+R9+R11+R12+R13+R14+R15+R16+
                 R17+R20+R22+R24, data = ct.valid.df, method = "class", 
               cp = 0.001, minsplit = 5, xval = 5, maxdepth=8)

# plot tree (cross-validation)
prp(cv.ct, type = 1, extra = 1,  split.font = 1, varlen = -10)

# prune
pruned.ct <- prune(cv.ct, 
                   cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])

# plot tree (pruned)
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10)  

#----------Naive Bayes-----------

new.bk <- bk.df[, c("D","R3","R9", "R11","R17","R23")]
new.bk.df <- data.frame(new.bk)
new.bk.df

# run naive Bayes
bk.nb <- naiveBayes(D~ ., data = new.bk.df)
bk.nb

# predict probabilities
probs.nb <- predict(bk.nb, newdata=new.bk.df, type='raw')
probs.nb

# predict class membership
class.nb <- predict(bk.nb, newdata = new.bk.df)
class.nb

# check model output
actual.prob<- data.frame(PRED_PROB_NB=new.bk.df$D)
new.bk.df$PRED_PROB_NB <-actual.prob
new.bk.df

# partition the data into training (60%) and validation (40%)
nb.train.index <- sample(c(1:dim(bk.df[,-c(1,3)])[1]), dim(bk.df[,-c(1,3)])[1]*0.6)  
nb.train.df <- bk.df[nb.train.index,]
nb.valid.df <- bk.df[-nb.train.index,]

# run naive Bayes on the training set
bk.total.nb <- naiveBayes(D~., data=nb.train.df)

# produce the confusion matrix - training
pred.class <- predict(bk.total.nb, newdata=nb.train.df)
confusionMatrix(pred.class, as.factor(nb.train.df$D))

# run naive Bayes on the training set
bk.total.nb2 <- naiveBayes(D~., data=nb.valid.df)

# produce the confusion matrix - validation
pred.class <- predict(bk.total.nb2, newdata=nb.valid.df)
NB.CM <- confusionMatrix(pred.class, as.factor(nb.valid.df$D))

# Conclusion

# Stepwise Regression Accuracy
SWR.CM

#KNN Accuracy
KNN.CM

# Classification Tree Accuracy
CT.CM

# Naive Bayes Accuracy
NB.CM
