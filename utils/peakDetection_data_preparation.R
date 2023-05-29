######### PEAK DETECTION ##########

gsub2<-function(pattern,replacement,x){
  for(i in 1:length(pattern))
    x<-gsub(pattern[i], replacement[i],x)
  x
}

# Function to Convert Timestamps to a suitable format
# Input is a dataframe containing the number of tweets for each timestamp in the original format ("2015-09-13 20:25:12.000Z")
# Output is a dataframe containing number of tweets for each timestamp in the POSIXlt format which can be used effectiely within R.

convertToTimestamp<-function(x){
  library(lubridate)
  library(date)
  from<-c("T",".000Z")
  to<-c(" ","")
  x$postedTime<-gsub2(from,to,x$postedTime)
  x$postedTime<-as.POSIXlt(x$postedTime,"GMT")
  return(x)
}

# Function to find the number of twets per day per hour
# Input: Dataframe containing the number of tweets for each timestamp
# Ouput: Dataframe with 3 attributes: Posted Date (YYYY-MM-DD), hour of the tweet (0-24) and Records (Number of Tweets)

getNumberOfTweetsPerDayPerHour<-function(x){
  library(DBI)
  library(RSQLite)
  library(proto)
  library(gsubfn)
  library(sqldf)
  library(chron)
  hour<-hour(x$postedTime)
  x<-cbind(x,hour)
  timestamp<-as.character(x$postedTime)
  x$postedTime<-as.chron(timestamp,"%Y-%m-%d %H:%M:%S")
  posted_date<-dates(x$postedTime)
  posted_date<-as.Date(posted_date,format="%Y-%m-%d")
  x<-cbind(x,posted_date)
  number_of_tweets_per_day_per_hour<-sqldf("select posted_date,hour,sum(Number_of_Tweets)Records from x group by 1,2")
  return(number_of_tweets_per_day_per_hour)
}