import csv
import re
import numpy
import math
import os
import time



class Script:
    my_path = os.path.abspath(os.path.dirname(__file__))

    # storage chi_square_score for all words of training data
    chi_square_score = os.path.join(my_path, "chi_square_score.txt")

    # storage the most 1000 important word to save time
    most_important = os.path.join(my_path, "most_important.txt")
    train_file = os.path.join(my_path, "trg.csv")
    test_file = os.path.join(my_path, "tst.csv")

    # storage result
    result_file = os.path.join(my_path, "result.csv")


# Use chi-square score to preprocess the trg.csv and get the 1000 most important
# words of 0~3600 rows of trg.csv and save them to most_important.txt
def dataPreprocess():
    print("Script started ... ")

    args_each = [None] * 3600  # storage number of words in each line
    args = set()  # storage all words, can filter out duplicate words.
    classes = {}  # storage classes and their count
    B_each_row = []
    A_each_row = []
    E_each_row = []
    V_each_row = []
    regexp = re.compile(r'^([^(?!a-z|A-Z)])*$')  # filter out all non-letter elements.
    with open(Script.train_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                line = 1
                continue
            if line == 3601:
                break
            if classes.keys().__contains__(row[1]):
                classes[row[1]] = classes[row[1]] + 1
            else:
                classes[row[1]] = 1
            document = row[2]  # the 3rd column is a document
            docs = document.split(r' ')

            # filter out if words are only-number, '' and any not includes letters
            real_docs = [doc for doc in docs if doc != '' and not regexp.search(doc) and not doc.isdigit()]
            for doc in real_docs:
                args.add(doc)  # collect all words
            wrapper = numpy.array(real_docs)
            unique, counts = numpy.unique(wrapper, return_counts=True)
            dic = dict(zip(unique, counts))
            if row[1] is 'B':
                B_each_row.append(dic)
            elif row[1] is 'V':
                V_each_row.append(dic)
            elif row[1] is 'A':
                A_each_row.append(dic)
            elif row[1] is 'E':
                E_each_row.append(dic)
            args_each[line - 1] = dic
            line = line + 1

    # now, all words are are recorded in the args,
    # and we need to statistic how many occurrence for each frequency for each words
    occurrence_frequency = {}
    chi_square_score = {}
    #  apply chi-square to filter the most 1000 important words
    for each in args_each:  # iterate each row of document
        for key in each:  # iterate each word's frequency
            if not occurrence_frequency.keys().__contains__(key):
                occurrence_frequency[key] = {}
            if occurrence_frequency[key].keys().__contains__(
                    each[key]):  # e.g. this row we found five 'the', so the frequency of '5-the' +1
                occurrence_frequency[key][each[key]] = occurrence_frequency[key][each[key]] + 1
            else:
                occurrence_frequency[key][each[key]] = 1

    # supply non-exist words as zero in each row.
    for of in occurrence_frequency.values():
        word_sum = 0
        for num in of.values():
            word_sum = word_sum + num
        of[0] = 3600 - word_sum

    for key in occurrence_frequency.keys():  # 'occurrence_frequency' storages the frequency of
        # all '0-the', '1-the', '2-the', '0-a', '1-a', ..., for all words
        word_and_of = occurrence_frequency[key]
        chi_score = 0
        for num in word_and_of.keys():
            xi = 0
            oijA = 0
            if num == 0:  # e.g., '0-the', if this row of A class has no '0-the' then the frequency of '0-the' + 1
                for row in A_each_row:
                    if key not in row:
                        oijA = oijA + 1
            else:
                for row in A_each_row:
                    if key in row and num == row[key]:
                        oijA = oijA + 1
            eijA = (word_and_of[num] / 3600) * (classes['A'] / 3600) * 3600
            xi = (oijA - eijA) ** 2 / eijA
            oijB = 0
            if num == 0:
                for row in B_each_row:
                    if key not in row:
                        oijB = oijB + 1
            else:
                for row in B_each_row:
                    if key in row and num == row[key]:
                        oijB = oijB + 1
            eijB = (word_and_of[num] / 3600) * (classes['B'] / 3600) * 3600
            xi = xi + (oijB - eijB) ** 2 / eijB
            oijV = 0
            if num == 0:
                for row in V_each_row:
                    if key not in row:
                        oijV = oijV + 1
            else:
                for row in V_each_row:
                    if key in row and num == row[key]:
                        oijV = oijV + 1
            eijV = (word_and_of[num] / 3600) * (classes['V'] / 3600) * 3600
            xi = xi + (oijV - eijV) ** 2 / eijV
            oijE = 0
            if num == 0:
                for row in E_each_row:
                    if key not in row:
                        oijE = oijE + 1
            else:
                for row in E_each_row:
                    if key in row and num == row[key]:
                        oijE = oijE + 1
            eijE = (word_and_of[num] / 3600) * (classes['E'] / 3600) * 3600
            xi = xi + (oijE - eijE) ** 2 / eijE
            chi_score = chi_score + xi
        chi_square_score[key] = chi_score

    chi_square_score = {k: v for k, v in sorted(chi_square_score.items(), key=lambda item: item[1], reverse=True)}
    f = open(Script.chi_square_score, "a")  # save the max 1000 words into most_important.txt.
    for chi_key in chi_square_score.keys():
        f.write(str(chi_key) + " : " + str(chi_square_score[chi_key]) + '\n')
    f.close()
    most_important = [0] * 1000
    index = 0
    for key in chi_square_score.keys():
        most_important[index] = key
        if index == 999:
            break
        else:
            index = index + 1

    f = open(Script.most_important, "a")
    for word in most_important:
        f.write(str(word) + '\n')
    f.close()
    return most_important


def NaiveBayesClassifier(most_important, validation_set, ground_truth,  is_validation):
    # save all Conditional Probabilities for all classes
    cp_a = {}
    cp_e = {}
    cp_v = {}
    cp_b = {}

    # get all words count on each row
    args = set()  # storage all words, can filter out duplicate words.
    B_each_row = []
    A_each_row = []
    E_each_row = []
    V_each_row = []
    regexp = re.compile(r'^([^(?!a-z|A-Z)])*$')  # filter out all non-letter elements.
    with open(Script.train_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                line = 1
                continue
            if line >= 3601:
                break
            document = row[2]  # the 3rd column is a document
            docs = document.split(r' ')

            # filter out if words are only-number, '' and any not includes letters
            real_docs = [doc for doc in docs if doc != '' and not regexp.search(doc) and not doc.isdigit()]
            for doc in real_docs:
                args.add(doc)  # collect all words
            wrapper = numpy.array(real_docs)
            unique, counts = numpy.unique(wrapper, return_counts=True)
            dic = dict(zip(unique, counts))
            if row[1] is 'B':
                B_each_row.append(dic)
            elif row[1] is 'V':
                V_each_row.append(dic)
            elif row[1] is 'A':
                A_each_row.append(dic)
            elif row[1] is 'E':
                E_each_row.append(dic)
            line = line + 1

    # count words number in each class
    words_num_a = 0
    words_num_b = 0
    words_num_e = 0
    words_num_v = 0

    # how many classes
    num_a = len(A_each_row) / 3600
    num_b = len(B_each_row) / 3600
    num_v = len(V_each_row) / 3600
    num_e = len(E_each_row) / 3600

    # count word number in each class
    for row in A_each_row:
        for key in row.keys():
            if key in most_important:
                words_num_a = words_num_a + row[key]

    for row in B_each_row:
        for key in row.keys():
            if key in most_important:
                words_num_b = words_num_b + row[key]

    for row in E_each_row:
        for key in row.keys():
            if key in most_important:
                words_num_e = words_num_e + row[key]

    for row in V_each_row:
        for key in row.keys():
            if key in most_important:
                words_num_v = words_num_v + row[key]

    version_space = len(most_important)

    normalised_count_A = [0] * len(A_each_row)  # transform based on length example in class A
    normalised_count_B = [0] * len(B_each_row)  # transform based on length example in class B
    normalised_count_E = [0] * len(E_each_row)  # transform based on length example in class E
    normalised_count_V = [0] * len(V_each_row)  # transform based on length example in class V

    index = 0
    for row in A_each_row:
        normalised_count = 0
        for key in row.keys():
            if key in most_important:
                normalised_count = normalised_count + row[key] ** 2
        normalised_count = math.sqrt(normalised_count)
        normalised_count_A[index] = normalised_count
        index = index + 1

    index = 0
    for row in B_each_row:
        normalised_count = 0
        for key in row.keys():
            if key in most_important:
                normalised_count = normalised_count + row[key] ** 2
        normalised_count = math.sqrt(normalised_count)
        normalised_count_B[index] = normalised_count
        index = index + 1

    index = 0
    for row in E_each_row:
        normalised_count = 0
        for key in row.keys():
            if key in most_important:
                normalised_count = normalised_count + row[key] ** 2
        normalised_count = math.sqrt(normalised_count)
        normalised_count_E[index] = normalised_count
        index = index + 1

    index = 0
    for row in V_each_row:
        normalised_count = 0
        for key in row.keys():
            if key in most_important:
                normalised_count = normalised_count + row[key] ** 2
        normalised_count = math.sqrt(normalised_count)
        normalised_count_V[index] = normalised_count
        index = index + 1

    # count likelihood for words in each class
    for word in most_important:
        num = 0
        index = 0
        for row in A_each_row:
            if row.keys().__contains__(word):
                # num = num + (row[word] / normalised_count_A[index])  # downweight by the word occurrence in this document.
                num = num + (row[word] / 1)  # downweight by the word occurrence in this document.
            index = index + 1
        cp_each = (num + 1) / (words_num_a + version_space)
        cp_a[word] = cp_each

        num = 0
        index = 0
        for row in B_each_row:
            if row.keys().__contains__(word):
                num = num + (row[word] / 1)  # downweight by the word occurrence in this document.
            index = index + 1
        cp_each = (num + 1) / (words_num_b + version_space)
        cp_b[word] = cp_each

        num = 0
        index = 0
        for row in E_each_row:
            if row.keys().__contains__(word):
                num = num + (row[word] / 1)  # downweight by the word occurrence in this document.
            index = index + 1
        cp_each = (num + 1) / (words_num_e + version_space)
        cp_e[word] = cp_each

        num = 0
        index = 0
        for row in V_each_row:
            if row.keys().__contains__(word):
                num = num + (row[word] / 1)  # downweight by the word occurrence in this document.
            index = index + 1
        cp_each = (num + 1) / (words_num_v + version_space)
        cp_v[word] = cp_each

    validation_result = [0] * len(validation_set)

    index = 0
    for row in validation_set:
        # initial all conditional probabilities by class proportion
        p_a = math.log10(num_a)
        p_e = math.log10(num_e)
        p_b = math.log10(num_b)
        p_v = math.log10(num_v)
        # p_a = num_a
        # p_e = num_e
        # p_b = num_b
        # p_v = num_v
        wrapper = numpy.array(row.split(' '))
        unique, counts = numpy.unique(wrapper, return_counts=True)
        dic = dict(zip(unique, counts))
        normalised_count = 0  # transform based on length example
        candidates = []  # save all candidates in this row
        for cp_word in dic.keys():
            if cp_word in most_important:
                # normalised_count = normalised_count + dic[cp_word]**2
                candidates.append(cp_word)
        # normalised_count = math.sqrt(normalised_count)
        for cp_word in candidates:
            # multiple size of candidate words to avoid too small to overflow
            # p_a = p_a * ((cp_a[cp_word] ** dic[cp_word])/1) * len(most_important)
            # p_b = p_b * ((cp_b[cp_word] ** dic[cp_word])/1) * len(most_important)
            # p_v = p_v * ((cp_v[cp_word] ** dic[cp_word])/1) * len(most_important)
            # p_e = p_e * ((cp_e[cp_word] ** dic[cp_word])/1) * len(most_important)
            # Use multi-nominal NBC and log the probabilities
            p_a = p_a + dic[cp_word] * math.log10(cp_a[cp_word])
            p_b = p_b + dic[cp_word] * math.log10(cp_b[cp_word])
            p_v = p_v + dic[cp_word] * math.log10(cp_v[cp_word])
            p_e = p_e + dic[cp_word] * math.log10(cp_e[cp_word])
        if p_a > p_e and p_a > p_b and p_a > p_v:
            validation_result[index] = 'A'
        if p_b > p_e and p_b > p_a and p_b > p_v:
            validation_result[index] = 'B'
        if p_v > p_a and p_v > p_b and p_v > p_e:
            validation_result[index] = 'V'
        if p_e > p_a and p_e > p_b and p_e > p_v:
            validation_result[index] = 'E'
        index = index + 1

    if is_validation:  # validation process
        correct_num = 0
        index = 0
        wrong_indexs = []
        for predict in validation_result:
            if predict == ground_truth[index]:
                correct_num = correct_num + 1
            else:
                wrong_indexs.append(index)
            index = index + 1
        print('correct rate of validation set is: ' + str(correct_num / len(ground_truth) * 100) + "%")
    else:  # test process
        print(validation_result)
        f = open(Script.result_file, "w")
        f.write("id,class\n")
        index = 1
        for result in validation_result:
            f.write(str(index) + "," + result + '\n')
            index = index + 1
        f.close()


if __name__ == '__main__':
    # if you want to run dataPreprocess function, please ensure "Script.train_file"
    # is pointing to trg.csv; and dataPreprocess function does not support other
    # data set but only trg.csv.

    # words = dataPreprocess()  # get the most 1000 important words, use chi-square score to select

    words = []
    f = open(Script.most_important, "r")
    for word in f:
        words.append(word.split('\n')[0])
    f.close()

    validation_start = 0
    validation_set = []  # read validation set begin at 3601
    ground_truth = []
    with open(Script.train_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if validation_start > 3600:  # use the 3601st-4000th data of trg.csv as a validation set
                validation_set.append(row[2])
                ground_truth.append(row[1])
            else:
                validation_start = validation_start + 1
                continue

    # Please set right path to "Script.train" variable, before evaluate your data,
    # which is the trg.csv; and please load your test data into a
    # List object, and pass it into method "NaiveBayesClassifier()" as a second
    # parameter, and please set the ground true list to be the third parameter
    # and the fourth parameter please set to be True
    startTime = time.time()
    NaiveBayesClassifier(words, validation_set, ground_truth, True)
    endTime = time.time()
    print("process time is " + str(endTime - startTime) + " seconds")

    # test_set = []  # read test set
    # with open(Script.test_file) as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     index = 0
    #     for row in csv_reader:
    #         if index == 0:  # skip first line
    #             index = 1
    #             continue
    #         test_set.append(row[1])
    #
    # if you want to test a dataset, please first load into a
    # list and then set to be second parameter; the third and
    # the fourth parameters please set them to be None and False
    # NaiveBayesClassifier(words, test_set, None,  False)
