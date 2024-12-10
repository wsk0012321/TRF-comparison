library(dplyr)
library(ggplot2)

readtables <- function(directory) {
  
  frame = data.frame()
  # load and append all the tables
  for (path in list.files(directory)) {
    file = paste0(directory,path)
    data = read.csv(file)
    # extend the columnes
    keys = substr(path,0,nchar(path)-3)
    parameters = strsplit(keys,split = '_')
    scalar = rep(parameters[[1]][1],8)
    embedding = rep(parameters[[1]][2],8)
    formula = rep(parameters[[1]][3],8)
    data = cbind(data,scalar,embedding,formula)
    # append to the dataframe
    frame = rbind(frame,data)
  }
  
  return(frame)
} 

directory <- "E:/PhD/data/Di_Liberto/transformed/Natural Speech/statistics/"
df <- readtables(directory)

aov_Cz <- aov(Cz ~ scalar * embedding * formula, data=df)
summary(aov_Cz)

aov_Pz <- aov(Cz ~ scalar * embedding * formula, data=df)
summary(aov_Cz)

aov_Fz <- aov(Cz ~ scalar * embedding * formula, data=df)
summary(aov_Cz)
