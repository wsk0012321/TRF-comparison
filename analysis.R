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

conversion <- function(df) {
  mean_acc = c()
  for (i in 1:nrow(df)) {
    sum_acc = 0
    for (n in 1:3) {
      if (df[i,n] < 0) {
        df[i,n] = abs(df[i,n]) # convert each negative value to a positive one
      }
      sum_acc = sum_acc + df[i,n]
    }
    mean_acc = c(mean_acc,sum_acc/3) # compute the mean accuracy over three channels
  }
  df = cbind(df,mean_acc)
  
  return(df)
}

directory <- "E:/PhD/data/Di_Liberto/transformed/Natural Speech/statistics/"
df <- readtables(directory)
df <- conversion(df)

# channel Cz
aov_Cz <- aov(Cz ~ scalar * embedding * formula, data=df)
summary(aov_Cz)
# channel Pz
aov_Pz <- aov(Pz ~ scalar * embedding * formula, data=df)
summary(aov_Cz)
# channel Fz
aov_Fz <- aov(Fz ~ scalar * embedding * formula, data=df)
summary(aov_Cz)
# averaged
aov_mean <- aov(mean_acc ~ scalar * embedding * formula, data=df)
summary(aov_mean)

# effect of scalar
df %>%
  ggplot(aes(x=scalar,y=mean_acc)) +
  geom_boxplot()+
  geom_point()+
  theme_bw()+
  labs(title='Effect of scalar',x='scalar',y='accuracy')

# effect of word embedding
df %>%
  ggplot(aes(x=embedding,y=mean_acc)) +
  geom_boxplot()+
  geom_point()+
  theme_bw()+
  labs(title='Effect of word embedding',x='scalar',y='accuracy')

# effect of formula
df %>%
  ggplot(aes(x=formula,y=mean_acc)) +
  geom_boxplot()+
  geom_point()+
  theme_bw()+
  labs(title='Effect of similarity metrics',x='scalar',y='accuracy')
