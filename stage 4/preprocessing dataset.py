from sklearn.feature_extraction.text import CountVectorizer
from unicodedata import normalize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
from autocorrect import spell
vect = TfidfVectorizer(stop_words='english',max_features=50)
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
tokenizer=nltk.tokenize.TreebankWordTokenizer()
stemmer=nltk.stem.PorterStemmer()
lemmaa=nltk.stem.WordNetLemmatizer()
path="/Users/Mohammed/Desktop/hellow/jordan1.txt"
file=open(path,"r")
messages=file.readlines()
print(messages)
spellchecker=[]
outofdigit=[]
toknization=[]
stemmessages=[]
filterdmessgaes=[]
deletexe=[]
deletehttp=[]
print (len(messages))
for i in range (len(messages)):
    toknization.append(re.sub(r'[0-9]',"", messages[i]))
    filterdmessgaes.append(re.sub(r'\\xa',"", toknization[i]))
    deletexe.append(re.sub(r'\\xc', "", filterdmessgaes[i]))
    deletehttp.append(re.sub(r'\\xe', "", deletexe[i]))
    outofdigit.append(re.sub(r'http', "", deletehttp[i]))
print(outofdigit)
'''for i in range (len(outofdigit)):
    tokens=tokenizer.tokenize(outofdigit[i])
    spellchecker.append(' '.join(spell(token) for token in tokens) + '\n')'''

for i in range (len(outofdigit)):
    tokens=tokenizer.tokenize(outofdigit[i])

    stemmessages.append(' '.join(stemmer.stem(token) for token in tokens)+'\n')

#print(tokens)
#print(stemmessages)
file2=open('/Users/Mohammed/Desktop/hellow/new.csv','w')
file2.writelines(stemmessages)
file3=open('/Users/Mohammed/Desktop/hellow/stem.txt','w')
file3.writelines(stemmessages)
file2=open('/Users/Mohammed/Desktop/hellow/new.csv','r')
newmessages=file2.readlines()
#print(newmessages)
jo=[]
for i in range (len(newmessages)):
    tokens = tokenizer.tokenize(newmessages[i])
    jo.append(" ".join(w for w in tokens if not w in stop_words)+'\n')
vect.fit(jo)
f=vect.get_feature_names()
print(f)
dtm = vect.transform(jo)
print(dtm)
repr(dtm)
df=pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
df.to_csv('/Users/Mohammed/Desktop/hellow/new.csv')



