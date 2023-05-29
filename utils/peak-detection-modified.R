# Collection of Peak Detection algorithms for us to evaluate
#
# Philip Healy (p.healy@cs.ucc.ie)

library("splus2R")
library("ifultools")
library("MASS")
library("sets")
library("wmtsa")

source("plots.R")

evaluateDetectionFunction <- function(datasets, f, fparams, plot=FALSE) {
  result <- list()
  result[["peak-function"]] <- peakFunctionCallToString(substitute(f), fparams)
  truePeaksListForTotal <- list()
  detectedPeaksListForTotal <- list()
  for (dataset in datasets) {
    truePeaks <- dataset[["truePeaks"]]
    fparams[["vec"]] <- dataset[["tweetVolumes"]]
    detectedPeaks <- do.call(f, fparams)
    
    #print(paste(dataset[["code"]],"true peaks"))
    #print(str(truePeaks))
    #print(paste(dataset[["code"]],"detected peaks"))
    #print(str(detectedPeaks))
    
    truePeaksListForTotal[[dataset[["code"]]]] <- truePeaks
    detectedPeaksListForTotal[[dataset[["code"]]]] <- detectedPeaks
    
    numElements <- length(dataset[["tweetVolumes"]])
    
    truePositives <- intersect(truePeaks, detectedPeaks)
    falsePositives <- setdiff(detectedPeaks, truePeaks)
    falseNegatives <- setdiff(truePeaks, detectedPeaks)
    
    #print(paste(dataset[["code"]],"true positives"))
    #print(str(length(truePositives)))
    #print(paste(dataset[["code"]],"false positives"))
    #print(str(length(falsePositives)))
    #print(paste(dataset[["code"]],"false negatives"))
    #print(str(length(falseNegatives)))
    
    if (plot) {
      tweetVolumePlotWithAlgorithmOutput(
        result[["peak-function"]], dataset, truePositives,
        falsePositives, falseNegatives)
    }
    
    precision <- calculatePrecision(truePeaks, detectedPeaks)
    recall <- calculateRecall(truePeaks, detectedPeaks)
    f1 <- calculateF1(precision, recall)
    
    result[[paste(dataset[["code"]], "elements", sep="-")]] <- numElements
    result[[paste(dataset[["code"]], "precision", sep="-")]] <- precision
    result[[paste(dataset[["code"]], "recall", sep="-")]] <- recall
    result[[paste(dataset[["code"]], "f1", sep="-")]] <- f1
  }
  overallPrecision <- calculateOverallPrecision(truePeaksListForTotal, detectedPeaksListForTotal)
  overallRecall <- calculateOverallRecall(truePeaksListForTotal, detectedPeaksListForTotal)
  overallF1 <- calculateF1(overallPrecision, overallRecall)
  
  result[["overall-precision"]] <- overallPrecision
  result[["overall-recall"]] <- overallRecall
  result[["overall-f1"]] <- overallF1
  
  
  return(result)
}

peakFunctionCallToString <- function(fname, fparams) {
  output <- paste(fname, " ", sep="")
  for (fparamName in names(fparams)) {
    if (fparamName == "tweetVolumes") {
      next
    }
    else {
      output <- paste(output, fparamName, "=", fparams[[fparamName]], sep="")
      output <- paste(output, " ", sep="")
    }
  }
  return(output)
}

calculatePrecision <- function(truePeaks, detectedPeaks) {
  l1 <- length(intersect(truePeaks, detectedPeaks))
  l2 <- length(detectedPeaks)
  return(l1 / l2)
}

calculateRecall <- function(truePeaks, detectedPeaks) {
  l1 <- length(intersect(truePeaks, detectedPeaks))
  l2 <- length(truePeaks)
  return(l1 / l2)
}

calculateF1 <- function(precision, recall) {
  return((2*precision*recall)/(precision + recall))
}

calculateOverallPrecision <- function(truePeaksList, detectedPeaksList) {
  totalTruePeaks <- 0
  totalDetectedPeaks <- 0
  totalCorrectlyDetectedPeaks <- 0
  
  #print("In calculateOverallPrecision()")
  #print(str(truePeaksList))
  #print("Detected peaks")  
  #print(str(detectedPeaksList))
  print(length(truePeaksList))
  print(length(detectedPeaksList))
  
  if(length(detectedPeaksList)>0){
  for (i in 1:length(detectedPeaksList)) {
    truePeaks <- truePeaksList[[i]]
    detectedPeaks <- detectedPeaksList[[i]]
    correctlyDetectedPeaks <- intersect(truePeaks, detectedPeaks)
    
    totalTruePeaks <- totalTruePeaks + length(truePeaks)
    totalDetectedPeaks <- totalDetectedPeaks + length(detectedPeaks)
    totalCorrectlyDetectedPeaks <- totalCorrectlyDetectedPeaks + length(correctlyDetectedPeaks)
    }
    return(totalCorrectlyDetectedPeaks / totalDetectedPeaks)
  }
  else
  {
    for (i in 1:length(truePeaksList)) {
    totalTruePeaks <- totalTruePeaks + length(truePeaks)
    totalCorrectlyDetectedPeaks<-0
    }
    return(0)
  }
}

calculateOverallRecall <- function(truePeaksList, detectedPeaksList) {  
  totalTruePeaks <- 0
  totalDetectedPeaks <- 0
  totalCorrectlyDetectedPeaks <- 0
  if(length(detectedPeaksList)>0){
  for (i in 1:length(detectedPeaksList)) {
    truePeaks <- truePeaksList[[i]]
    detectedPeaks <- detectedPeaksList[[i]]
    correctlyDetectedPeaks <- intersect(truePeaks, detectedPeaks)
    
    totalTruePeaks <- totalTruePeaks + length(truePeaks)
    totalDetectedPeaks <- totalDetectedPeaks + length(detectedPeaks)
    totalCorrectlyDetectedPeaks <- totalCorrectlyDetectedPeaks + length(correctlyDetectedPeaks)
    }
  return(totalCorrectlyDetectedPeaks / totalTruePeaks)
  }
  else{
    for (i in 1:length(truePeaksList)) {
    totalTruePeaks <- totalTruePeaks + length(truePeaks)
    totalCorrectlyDetectedPeaks<-0
    }
    return(0)
  }
}

# Wavelet method from wmtsa library
# Calculates the CWT and peaks using default scale.range
findPeaksCWT <- function(vec) {
  W <- wavCWT(vec)
  z <- wavCWTTree(W)
  p <- wavCWTPeaks(z)
  return(p[[1]])
}

# Timothee Poissot's Extrema algorithm, see
# http://rtricks.wordpress.com/2009/05/03/an-algorithm-to-find-local-extrema-in-a-vector/
# Window width is 2*bw + 1
findPeaksPoissot <- function(vec,bw=12,x.coo=c(1:length(vec)))
{
  pos.x.max <- NULL
  pos.y.max <- NULL
  pos.x.min <- NULL
  pos.y.min <- NULL
  for(i in 1:(length(vec)-1))   {
    if((i+1+bw)>length(vec)){
      sup.stop <- length(vec)}
    else{
      sup.stop <- i+1+bw
    }
    if((i-bw)<1){
      inf.stop <- 1
    }
    else{
      inf.stop <- i-bw
    }
    subset.sup <- vec[(i+1):sup.stop]
    subset.inf <- vec[inf.stop:(i-1)]
    
    is.max   <- sum(subset.inf > vec[i]) == 0
    is.nomin <- sum(subset.sup > vec[i]) == 0
    
    no.max   <- sum(subset.inf > vec[i]) == length(subset.inf)
    no.nomin <- sum(subset.sup > vec[i]) == length(subset.sup)
    
    if(is.max & is.nomin){
      pos.x.max <- c(pos.x.max,x.coo[i])
      pos.y.max <- c(pos.y.max,vec[i])
    }
    if(no.max & no.nomin){
      pos.x.min <- c(pos.x.min,x.coo[i])
      pos.y.min <- c(pos.y.min,vec[i])
    }
  }
  #return(list(pos.x.max,pos.y.max,pos.x.min,pos.y.min))
  return(data.frame(hour=pos.x.max, tweets=pos.y.max))
}


palshikarS1 <- function(k, i, xi, t)
{
  N <- length(t)
  firstLeftNeighbourIndex <- max(0, i - k)
  lastLeftNeighbourIndex <- max(0, i - 1)
  firstRightNeighbourIndex <- min(i + 1, N)
  lastRightNeighbourIndex <- min(i + k, N)
  
  leftNeighbours <- t[firstLeftNeighbourIndex:lastLeftNeighbourIndex]
  rightNeighbours <- t[firstRightNeighbourIndex:lastRightNeighbourIndex]
  
  if (length(leftNeighbours) == 0)
  {
    leftDiffMax <- 0
  }
  else
  {
    leftDiffMax <- max(xi - leftNeighbours)
  }
  
  if (length(rightNeighbours) == 0)
  {
    rightDiffMax <- 0
  }
  else
  {
    rightDiffMax <- max(xi - rightNeighbours)
  }
  
  return((leftDiffMax + rightDiffMax) / 2)
}

palshikarS4 <- function(k, i, xi, t)
{
  N <- length(t)
  firstLeftNeighbourIndex <- max(0, i - k)
  lastLeftNeighbourIndex <- max(0, i - 1)
  firstRightNeighbourIndex <- min(i + 1, N)
  lastRightNeighbourIndex <- min(i + k, N)
  
  leftNeighbours <- t[firstLeftNeighbourIndex:lastLeftNeighbourIndex]
  rightNeighbours <- t[firstRightNeighbourIndex:lastRightNeighbourIndex]
  
  centerAndNeighbours <- c(leftNeighbours, t[i], rightNeighbours)
  justNeighbours <- c(leftNeighbours, rightNeighbours)
  
  centerAndNeighboursEntropy <- entropy(centerAndNeighbours)
  justNeighboursEntropy <- entropy(justNeightbours)
  
  return(0.0)
}

entropy <- function(t) {
  return(0)
}

# Generalized Palshikar peak detection function
findPeaksPalshikar <- function(t, k, h, s)
{
  # Get significant function values for each xi in t
  N <- length(t)
  a <- rep(NA, N-1)
  for(i in 1:(N-1))
  {
    a[i] <- s(k, i, t[i], t)
  }
  
  # Get mean and standard deviation of positive values of a
  m <- mean(subset(a, a > 0))
  s <- sd(subset(a, a > 0))
  hour <- vector()
  tweets <- vector();
  for(i in 1:(N-1))
  {
     
    if ((a[i] > 0) & ((a[i] - m) > h*s))
    {
      hour <- c(hour, i)
      tweets <- c(tweets, t[i])
    }
  }
  
  # Remove adjacent peaks
  unfilteredOutput <- data.frame(hour, tweets)  
  adjacencyGroup <- data.frame(hour=c(), tweets=c())
  filteredOutput <- data.frame(hour=c(), tweets=c())
  for (i in 1:(nrow(unfilteredOutput) - 1))
  {
    adjacencyGroup <- rbind(adjacencyGroup, unfilteredOutput[i:i,])
    if ((hour[i + 1] - hour[i]) > k)
    {
      # Get max row in adjacency group and add to output
      maxNumTweetsInGroup <- max(adjacencyGroup$tweets)
      maxRows <- subset(adjacencyGroup, tweets == maxNumTweetsInGroup)
      filteredOutput <- rbind(filteredOutput, maxRows[1:1,])
      
      # Reset adjacency group
      adjacencyGroup <- data.frame(hour=c(), tweets=c())
    }
  }
  return(filteredOutput)
}

findPeaksPalshikarS1 <- function(vec)
{
  outputDf <- findPeaksPalshikar(vec, k=12, h=3, palshikarS1)
  print("Palshikar S1 output")
  print(str(outputDf))
  return(outputDf$hour)
}

findPeaksPalshikarS4 <- function(vec)
{
  outputDf <- findPeaksPalshikar(vec, k=3, h=1, palshikarS4)
  print("Palshikar S4 output")
  print(str(outputDf))
  return(outputDf$hour)
}

# Filter peaks so that we only return peaks with values above the specified
# percentile of peaks
filterByPeakPercentile <- function(peaks, percentile) {
  cutoff <- quantile(peaks$tweets, percentile)
  return(subset(peaks, peaks$tweets >= cutoff))
}

# Filter peaks so that we only return peaks with values above the specified
# percentile of input values
filterBySamplePercentile <- function(vec, peaks, percentile) {
  cutoff <- quantile(vec, percentile)
  return(subset(peaks, peaks$tweets >= cutoff))
}

findPeaksLehmann <- function(vec)
{
  peaks <- vector(mode="integer", length=0)
  for (i in 1:length(vec)) {
    if (lehmannIteration(vec, i)) {
      peaks <- c(peaks, i)
    }
  }
  peaks <- prunePeaksLehmann(peaks)
  print(str(peaks))
  return(peaks)
}

lehmannIteration <- function(vec, i)
{
  N <- length(vec)
  L <- 30
  windowStartIndex <- max(0, i - L)
  windowEndIndex <- min(i + L, N)
  window <- vec[windowStartIndex:windowEndIndex]
  currentActivity <- vec[i]
  medianActivity <- median(window)
  minActivity <- 10
  outlierFraction <-
    (currentActivity - medianActivity) /
    max(medianActivity, minActivity)
  threshold <- 10
  return(outlierFraction > threshold)
}

prunePeaksLehmann <- function(peaks)
{
  minGap <- 7
  if (length(peaks) < 2) {
    return(peaks)
  }
  else {
    lastPeakIndex <- peaks[1]
    prunedPeaks <- c(lastPeakIndex)
    for (i in 2:length(peaks)) {
      if ((peaks[i] - lastPeakIndex) > minGap) {
        prunedPeaks <- c(prunedPeaks, peaks[i])
        lastPeakIndex <- peaks[i]
      }
    }
    return(prunedPeaks)
  }
}