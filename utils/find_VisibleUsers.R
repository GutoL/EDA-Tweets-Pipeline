args <- commandArgs(trailingOnly = TRUE)

#import CSV with replies from the WD
replies<-read.csv(paste(args[1],"replies.csv",sep = ""))

# Constructing Reply Network
# Input: DataFrame with User Screennames and Tweets in that order
# Output: Dataframe with Source and Target. Source is the user who replied to a tweet
# and Target is the user whose tweets was replied to

constructReplyNetwork<-function(df){
  library(stringr)
  library(sqldf)
  names(df)<-c("userscreenname","body")
  df$body<-gsub("[^[:alnum:][:space:]_]", "", df[,2])
  df$reply_to_screen_name<-word(as.character(df$body),1)
  df$reply_to_screen_name<-substr(as.character(df$reply_to_screen_name),1,length(df$reply_to_screen_name)+1)
  reply_network_table<-df[,c("userscreenname","reply_to_screen_name")]
  reply_network<-sqldf("select userscreenname Source,reply_to_screen_name Target from reply_network_table where userscreenname !='' AND reply_to_screen_name !=''")
  reply_network<-unique(reply_network)
  return(reply_network)
  #write.csv(reply_network,"Reply Network.csv")
}

reply_network<-constructReplyNetwork(replies)

# Listing the Users along with the number of  replies they received

repliedto_users<-sqldf("select Target, count(Source) Number_of_Replies_Received from reply_network group by 1 order by 2 desc")
rm(replies, reply_network, constructReplyNetwork)

#import CSV with retweets from the WD
retweets<-read.csv(paste(args[1],"retweets.csv",sep = ""))

# Constructing the Retweet Network
# Similar to constructing Reply Network
# Input: DataFrame with User Screennames and Retweets in that order
# Output: Dataframe with Source and Target. Source is the user who retweeted to a tweet
# and Target is the user whose tweets was retweeted to

constructRetweetNetwork<-function(df){
  library(stringr)
  library(sqldf)
  names(df)<-c("userscreenname","body")
  df$body<-gsub("[^[:alnum:][:space:]_]", "", df[,2])
  df$retweet_to_screen_name<-word(as.character(df$body),2)
  df$retweet_to_screen_name<-substr(as.character(df$retweet_to_screen_name),1,length(df$retweet_to_screen_name)+1)
  retweet_network_table<-df[,c("userscreenname","retweet_to_screen_name")]
  retweet_network<-sqldf("select userscreenname Source,retweet_to_screen_name Target from retweet_network_table where userscreenname !='' AND retweet_to_screen_name !=''")
  retweet_network<-unique(retweet_network)
  return(retweet_network)
  #write.csv(retweet_network,"Retweet Network.csv")
}

retweet_network<-constructRetweetNetwork(retweets)

# Listing the Users along with the number of  retweets they received

retweetedto_users<-sqldf("select Target, count(Source) Number_of_Retweets_Received from retweet_network group by 1 order by 2 desc")
rm(retweet_network, constructRetweetNetwork, retweets)


# Finding visible users based on replies and retweets
# Ensure that all the required dataframes and functions are present in the workspace
#Output: CSV file with Visible Users, Number of Replies Recieved,Number of Retweets Received
#  and the sum of both (Replies Received + Retweets Received)

findingVisibleUsers<-function(){
  visible_users<-merge(retweetedto_users,repliedto_users,by="Target",all.x = TRUE,all.y = TRUE)
  visible_users$Number_of_Retweets_Received[is.na(visible_users$Number_of_Retweets_Received)]<-0
  visible_users$Number_of_Replies_Received[is.na(visible_users$Number_of_Replies_Received)]<-0
  visible_users$Visibility<-visible_users$Number_of_Retweets_Received+visible_users$Number_of_Replies_Received
  visible_users<-visible_users[order(-visible_users$Visibility),]
  write.csv(visible_users,paste(args[1],"Visible_Users.csv",sep = ""),row.names = FALSE)
}
findingVisibleUsers()