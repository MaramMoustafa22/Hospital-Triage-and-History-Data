library(dplyr)
library(magrittr)
library(caret)

#disposition

df$disposition <- as.numeric(df$disposition == 'Admit')
df$gender <- factor(df$gender,levels = c("Female","Male"),labels = c(1,0))
df$ethnicity <- factor(df$ethnicity,levels = c("Hispanic or Latino","Non-Hispanic","Patient Refused","Unknown"),labels = c(0,1,2,3))
df$religion <- factor(df$religion,levels = c("None","Pentecostal","Catholic","Protestant","Christian","Other","Unknown","Baptist","Methodist","Episcopal","Jewish","Muslim","Jehovah's Witness"),labels = c(0,1,2,3,4,5,6,7,8,9,10,11,12))
df$maritalstatus <- factor(df$maritalstatus,levels = c("Single","Married","Widowed","Significant Other","Divorced","Legally Separated","Other","Unknown","Life Partner","Civil Union"),labels = c(0,1,2,3,4,5,6,7,8,9))
df$insurance_status <- factor(df$insurance_status,levels = c("Other","Commercial","Medicare","'Medicaid","Self pay"),labels = c(0,1,2,3,4))
df$previousdispo <- factor(df$previousdispo,levels = c("No previous dispo","Discharge","Admit","Transfer to Another Facility","AMA","LWBS after Triage","Eloped","LWBS before Triage","Observation","Send to L&D"),labels = c(0,1,2,3,4,5,6,7,8,9))
df$dep_name <- factor(df$dep_name,levels = c("C","B","A"),labels = c(0,1,2))
df$arrivalmode <- factor(df$arrivalmode,levels = c("ambulance","Car","Other","Police","Public Transportation","Walk-in","Wheelchair"),labels = c(0,1,2,3,4,5,6))

df <- select(df,-c("employstatus","race","lang","arrivalmonth","arrivalday","arrivalhour_bin"))
df <- select(df,-c("esi"))
df$gender <- as.numeric(as.character(df$gender))
df$ethnicity <- as.numeric(as.character(df$ethnicity))
df$religion <- as.numeric(as.character(df$religion))
df$maritalstatus <- as.numeric(as.character(df$maritalstatus))
df$insurance_status <- as.numeric(as.character(df$insurance_status))
df$previousdispo <- as.numeric(as.character(df$previousdispo))
df$dep_name <- as.numeric(as.character(df$dep_name))
df$arrivalmode <- as.numeric(as.character(df$arrivalmode))
df[is.na(df)] <- 0

library(xgboost)
library(ranger)   
library(rpart)    
tree <- rpart(disposition ~ ., data = df)

(vi_tree <- tree$variable.importance)
View(vi_tree)
par(mar=c(3, 15, 3, 1))
barplot(vi_tree, horiz = TRUE, las = 1)


bst <- xgboost(
  data = data.matrix(subset(df, select = -disposition)),
  label = df$disposition, 
  objective = "reg:linear",
  nrounds = 100, 
  max_depth = 5, 
  eta = 0.3,
  verbose = 0  
)

(vi_bst <- xgb.importance(model = bst))
View(vi_bst)
xgb.ggplot.importance(vi_bst)

library(vip)
t <- vi(tree)
View(t)
g <- vi(bst)
View(g)

p1 <- vip(tree)
p2 <- vip(bst, aesthetics = list(col = "purple2")) 
grid.arrange(p1, p2, ncol = 2)
#run it on r
library(ggplot2) 
vip(bst, num_features = 13, horizontal = FALSE, 
    aesthetics = list(color = "red", size = 4)) +
  theme_light()


#save.image(file='myEnvironment.RData')