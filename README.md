# Naive-Bayes-classifier-pure-Python

Apply Chi-square score to filter out less important fields.

## Description

The goal of this project is using Python to implementation a Naive Bayes algorithm, in order to achieve a high predictive accuracy on the test data.

### Data
The data set containing the abstracts of research papers that deal with proteins. Each protein is found either in Archaea, Bacteria, Eukaryota, or Virus. The second attribute (class) in each record specifies the domain of the protein:

- A: Archaea
- B: Bacteria
- E: Eucaryota
- V: Virus

The third attribute (abstract) in the record is a character string containing the abstract to be classified. The string is preprocessed: it contains only whitespace, alphanumerical lowercase characters, digits, the dash (-) and the prime ('). Each contiguous sequence of non-whitespace characters framed by whitespace is considered a word.

The file [trg.csv](trg.csv) is the training set on which you have to train your Naive Bayes classifier. The file [tst.csv](tst.csv) is the test set (without the classes) for which we have to generate classifications and submit.

### Solution
First of all, we define the first 3600 lines of the [trg.csv](trg.csv) as the training set and the rest as validation set. Then we calculate Chi-Square score for every word in the training set and find the most 1000 important words; [chi_square_score.txt](chi_square_score.txt) records Chi-Square scores of all words in the training set and words in [most_important.txt](chi_square_score.txt) are the most 1000 important words; Finally, we apply multi-nominal NBC classifier to predict protein class. These algorithms can be found in [A5.py](A5.py), and the definition of multi-nominal NBC classifier can be found in [wiki](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_na%C3%AFve_Bayes).

### Result
Leveraging above techniques, the correct rate of class prediction on the test set [trg.csv](trg.csv) is **93.75%**, and the classification result can be found in [result.csv](result.csv)

