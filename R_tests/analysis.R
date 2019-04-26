setwd("C:/Users/CK/Documents/Works/AI2_SVN_BAYES2/R_tests")

data <- read.csv("data.csv", header =F)

word_names   <- scan(file = "names.txt", what="character")

spam <- data[data$V55 == 1, 1:54]
ham  <- data[data$V55 == 0, 1:54]

names(spam)<- word_names[1:54]
names(ham)<- word_names[1:54]


library('MVN')


spam_no_zeros <- spam[,colSums(spam!=0)>0]
ham_no_zeros  <- ham[, colSums(spam!=0)>0]

# Mardia's Test #

mvnSpam <- mvn(data = spam_no_zeros, mvnTest = "mardia")
mvnSpam$multivariateNormality

mvnHam <- mvn(data = ham_no_zeros, mvnTest = "mardia")
mvnHam$multivariateNormality


# Chi Squared #

suppressWarnings(chisq.test(spam$word_freq_free , spam$word_freq_business))
suppressWarnings(chisq.test(spam$word_freq_address, spam$word_freq_internet))
suppressWarnings(chisq.test(spam$word_freq_money, spam$word_freq_receive))



