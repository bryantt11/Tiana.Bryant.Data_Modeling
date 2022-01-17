#-------- Part A - Load the data into R Studio -------- 
#Loading the tidyverse package
library(tidyverse)

#Load CSV file using tidyverse
loanData <- read_csv("/Users/tianabryant/Downloads/loan.csv")

#Identify the structure
str(loanData)

#Identify rows/columns
df <- data.frame(loanData)
# How many rows and columns are there?
nrow(df)
ncol(df)

#Show top 6 rows of dataset
head(df, 6)

# -------- Part B - Create Visualizations in R Studio using ggplot() --------
#Create bar charts
library(ggplot2)
options(scipen=999)

#term
ggplot(data=df)+
  geom_bar(mapping=aes(x=term, fill=term))

#grade
ggplot(data=df)+
  geom_bar(mapping=aes(x=grade, fill=grade))

#emp_length
ggplot(data=df)+
  geom_bar(mapping=aes(x=emp_length, fill=emp_length))+
  coord_flip()

#loan_status
ggplot(data=df)+
  geom_bar(mapping=aes(x=loan_status, fill=loan_status))+
  coord_flip()

#home_ownership
ggplot(data=df)+
  geom_bar(mapping=aes(x=home_ownership, fill=home_ownership))

# additional visualizations
ggplot(data=df, mapping=aes(x=emp_length, fill=emp_length))+
  geom_bar()+
  geom_label(stat="count", mapping=aes(label=..count.., fill=emp_length))+
  coord_flip()

ggplot(data=df, mapping=aes(x=home_ownership, fill=home_ownership))+
  geom_bar()+
  geom_label(stat="count", mapping=aes(label=..count.., fill=home_ownership))

#Histogram on the loan amount
ggplot(data=df)+
  geom_histogram(mapping=aes(loan_amnt), binwidth=1000, color="black", fill="purple")

#add data labels
ggplot(data=df, mapping=aes(loan_amnt))+
  geom_histogram(binwidth=1000, color="black", fill="purple")+
  geom_label(stat="bin", binwidth=1000, aes(label=..count..))

#Subset data frame, amounts < $15,000, annual incomes < $75,000
subset1 <- subset(df,loan_amnt<=15000 & annual_inc<=75000)
head(subset1)

#Create a scatter plot that examines both the annual income and loan amounts
ggplot(data=subset1)+
  geom_point(mapping=aes(x=annual_inc, y=loan_amnt, color=purpose))

#New data frame - employment info
#id, emp_title, emp_length, and annual_inc
df$member_id<- NULL
df$loan_amnt<- NULL
df$funded_amnt<- NULL
df$funded_amnt_inv<- NULL
df$term<- NULL
df$int_rate<- NULL
df$installment<- NULL
df$grade<- NULL
df$sub_grade<- NULL
df$home_ownership<- NULL
df[1:4]

#show the first 10 rows
head(df[1:4], 10)

#Annual income, sorted desc
head(arrange(df[1:4],desc(annual_inc)), 10)