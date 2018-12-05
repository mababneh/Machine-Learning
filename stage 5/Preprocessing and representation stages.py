import nltk
nltk.download('punkt')
import pandas as pd
import unicodedata
import re
from textblob import TextBlob
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')
def main ():
    vect = TfidfVectorizer(stop_words='english',max_features=75)# Creating an object from TFidfvector class to represent the text as bag of word  with TF-idf values
    vect2=TfidfVectorizer(stop_words='english',max_features=75,ngram_range=(2,2))# Creating an object from TFidfvector class to represent the text as 2-grams with TF-idf values
    stemmer=nltk.stem.PorterStemmer()#creating an stemmer object from port stemmer class to convert the word to thier base form
    #Reading the dataset file for preprocessing step
    path1=input("\n Please Insert the path of data : \n")
    file=open(path1,'r')
    lines=file.read().split('\n')
    #Reading the stopwprds for preprocessing step
    file2=open("stopwords.txt",'r')
    stopwordss=file2.read().split('\n')
    file3=open('SEn.txt','w')
    list1=[]
    list4=[]
    def clean_question(question):
        # clean function to clean Comment text by removing links, special characters using simple regex statements
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])| (\w+:\ / \ / \S+)", " ", question).split())
    def remove_stopwords(question):
        # Removing stop words from list of stopwords which are collected from internet.
        b=TextBlob(question)
        tokens=b.words
        return " ".join(w for w in tokens if not w in stopwordss)
    def remove_non_ascii(question):
        #Removing NON_Ascci character
        words_non = []
        b = TextBlob(question)
        tokens = b.words
        for token in tokens:
            new_word = unicodedata.normalize('NFKD', token).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            words_non.append(new_word)
        return " ".join(words_non)
    def word_stemming(question):
        #stemming reduce the words to thier stem word
        words_list=[]
        b=TextBlob(question)
        tokens=b.words
        for token in tokens:
            words_list.append(stemmer.stem(token))
        return " ".join(words_list)
    def get_sentiment(question):
        # THis function is used to compute the sentiement anaylsis by using Textblob library
        analysis = TextBlob(clean_question(question))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            file3.write(str(1)+'\n')
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            file3.write(str(0)+'\n')
            return 'neutral'
        else:
            file3.write(str(-1)+'\n')
            return 'negative'

    list0=[]
    list00=[]
    senti_anaylsis=[]
    for i in range (len(lines)):
        list1.append(clean_question(lines[i]))
        list0.append(remove_non_ascii(list1[i]))
        list00.append(word_stemming(str(list0[i])))
        list4.append(remove_stopwords(list00[i]))

    list4=list4[:-1]
    for i in range (len(list4)):
        senti_anaylsis.append(get_sentiment(list4[i]))
    print("--------------------------------Sentiment Anaylsis is done!!!!------------------------------------------------")
    vect.fit(list4)
    f=vect.get_feature_names()
    print(f)
    dtm = vect.transform(list4)
    repr(dtm)
    df=pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
    df.to_csv('train1.csv')
    print("------------------------Bag of words with TF-IDF is already done!!!-----------------------------------")

    vect2.fit(list4)
    f1=vect2.get_feature_names()
    print(f1)
    dtm1 = vect2.transform(list4)
    repr(dtm1)
    df=pd.DataFrame(dtm1.toarray(), columns=vect2.get_feature_names())
    df.to_csv('train2.csv')
    print("------------------------2-grams with TF-IDF is already done!!!---------------------------------------")
if __name__ == "__main__":
    main()
