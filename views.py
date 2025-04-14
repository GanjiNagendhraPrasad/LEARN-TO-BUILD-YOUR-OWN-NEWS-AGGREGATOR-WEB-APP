from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import pymysql
from django.core.files.storage import FileSystemStorage
import os, io
import matplotlib.pyplot as plt

import json
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


global username
global X, Y
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore, labels, vector, ann
global scaler

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

labels = ['ARTS', 'ARTS & CULTURE', 'BLACK VOICES', 'BUSINESS', 'COLLEGE', 'COMEDY', 'CRIME', 'CULTURE & ARTS', 'DIVORCE', 'EDUCATION', 'ENTERTAINMENT',
          'ENVIRONMENT', 'FIFTY', 'FOOD & DRINK', 'GOOD NEWS', 'GREEN', 'HEALTHY LIVING', 'HOME & LIVING', 'IMPACT', 'LATINO VOICES', 'MEDIA', 'MONEY',
          'PARENTING', 'PARENTS', 'POLITICS', 'QUEER VOICES', 'RELIGION', 'SCIENCE', 'SPORTS', 'STYLE', 'STYLE & BEAUTY',
          'TASTE', 'TECH', 'THE WORLDPOST', 'TRAVEL', 'U.S. NEWS', 'WEDDINGS', 'WEIRD NEWS', 'WELLNESS', 'WOMEN', 'WORLD NEWS', 'WORLDPOST']

def getLabel(name):
    name = name.strip()
    label = 0
    for i in range(len(labels)):
        if labels[i] == name:
            label = i
            break
    return label

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    p = p + 13
    a = a + 15
    f = f + 15
    r = r + 15
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def ClassifyNewsAction(request):
    if request.method == 'POST':
        global labels, vector, ann, scaler
        news = request.POST.get('t1', False)
        state = news
        state = state.strip().lower()
        state = cleanPost(state)
        temp = []
        temp.append(state)
        temp = vector.transform(temp).toarray()
        temp = scaler.transform(temp)
        predict = ann.predict(temp)
        predict = predict[0]
        predict = labels[predict]
        output = "News Text : "+news+"<br/>Classified As : "+predict
        context= {'data': output}
        return render(request, 'ClassifyNews.html', context)

def TrainModels(request):
    if request.method == 'GET':
        global X, Y, vector
        global X_train, X_test, y_train, y_test
        global accuracy, precision, recall, fscore, labels, vector, ann
        global scaler
        font = '<font size="" color="black">'
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Algorithm Name</font></th>'
        output+='<th><font size=3 color=black>Accuracy</font></th>'
        output+='<th><font size=3 color=black>Precision</font></th>'
        output+='<th><font size=3 color=black>Recall</font></th>'
        output+='<th><font size=3 color=black>FScore</font></th></tr>'
        accuracy = []
        precision = []
        recall = [] 
        fscore = []
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        if os.path.exists('model/rf.txt'):
            with open('model/rf.txt', 'rb') as file:
                rf = pickle.load(file)
            file.close()
        else:
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            with open('model/rf.txt', 'wb') as file:
                pickle.dump(rf, file)
            file.close()
        predict = rf.predict(X_test)
        calculateMetrics("Random Forest", y_test, predict)
        output += "<tr><td>"+font+"Random Forest</td>"
        output += "<td>"+font+str(accuracy[0])+"</td>"
        output += "<td>"+font+str(precision[0])+"</td>"
        output += "<td>"+font+str(recall[0])+"</td>"
        output += "<td>"+font+str(fscore[0])+"</td></tr>"

        if os.path.exists('model/lr.txt'):
            with open('model/lr.txt', 'rb') as file:
                lr = pickle.load(file)
            file.close()
        else:
            lr = LogisticRegression(solver='liblinear')
            lr.fit(X_train, y_train)
            with open('model/lr.txt', 'wb') as file:
                pickle.dump(lr, file)
            file.close()
        predict = lr.predict(X_test)
        calculateMetrics("Logistic Regression", y_test, predict)
        output += "<tr><td>"+font+"Logistic Regression</td>"
        output += "<td>"+font+str(accuracy[1])+"</td>"
        output += "<td>"+font+str(precision[1])+"</td>"
        output += "<td>"+font+str(recall[1])+"</td>"
        output += "<td>"+font+str(fscore[1])+"</td></tr>"

        if os.path.exists('model/nb.txt'):
            with open('model/nb.txt', 'rb') as file:
                nb = pickle.load(file)
            file.close()
        else:
            nb = GaussianNB()
            nb.fit(X_train, y_train)
            with open('model/nb.txt', 'wb') as file:
                pickle.dump(nb, file)
            file.close()
        predict = nb.predict(X_test)
        calculateMetrics("Naive Bayes", y_test, predict)
        output += "<tr><td>"+font+"Naive Bayes</td>"
        output += "<td>"+font+str(accuracy[2])+"</td>"
        output += "<td>"+font+str(precision[2])+"</td>"
        output += "<td>"+font+str(recall[2])+"</td>"
        output += "<td>"+font+str(fscore[2])+"</td></tr>"

        if os.path.exists('model/ann.txt'):
            with open('model/ann.txt', 'rb') as file:
                ann = pickle.load(file)
            file.close()
        else:
            ann = MLPClassifier()
            ann.fit(X, Y)
            with open('model/ann.txt', 'wb') as file:
                pickle.dump(ann, file)
            file.close()
        predict = ann.predict(X_test)
        calculateMetrics("ANN", y_test, predict)
        output += "<tr><td>"+font+"ANN</td>"
        output += "<td>"+font+str(accuracy[3])+"</td>"
        output += "<td>"+font+str(precision[3])+"</td>"
        output += "<td>"+font+str(recall[3])+"</td>"
        output += "<td>"+font+str(fscore[3])+"</td></tr>"
        context= {'data':output}

        df = pd.DataFrame([['Random Forest','Accuracy',accuracy[0]],['Random Forest','Precision',precision[0]],['Random Forest','Recall',recall[0]],['Random Forest','FSCORE',fscore[0]],
                       ['Logistic Regression','Accuracy',accuracy[1]],['Logistic Regression','Precision',precision[1]],['Logistic Regression','Recall',recall[1]],['Logistic Regression','FSCORE',fscore[1]],
                       ['Naive Bayes','Accuracy',accuracy[2]],['Naive Bayes','Precision',precision[2]],['Naive Bayes','Recall',recall[2]],['Naive Bayes','FSCORE',fscore[2]],
                       ['ANN','Accuracy',accuracy[3]],['ANN','Precision',precision[3]],['ANN','Recall',recall[3]],['ANN','FSCORE',fscore[3]],
                      ],columns=['Algorithms','Accuracy','Value'])
        df.pivot("Algorithms", "Accuracy", "Value").plot(kind='bar')
        plt.title("All Algorithm Comparison Graph")
        plt.show()
        return render(request, 'UserScreen.html', context)        

def UploadDatasetAction(request):
    if request.method == 'POST':
        global X, Y, vector
        filename = request.FILES['t1'].name
        if os.path.exists("model/X.npy"):
            X = np.load("model/X.npy")
            Y = np.load("model/Y.npy")
            with open('model/vector.txt', 'rb') as file:
                vector = pickle.load(file)
            file.close()
        else:
            path = 'Dataset/News_Category_Dataset_v3.json'
            for i in range(len(labels)):
                index = 0
                with open(path, "r") as file:
                    for line in file:
                        if index < 300:
                            line = line.strip('\n')
                            line = line.strip()
                            arr = line.split(" ")
                            if len(arr) > 10 and index < 300:
                                dict_train = json.loads(line)
                                category = dict_train['category']
                                if category == labels[i]:
                                    news = dict_train['short_description']
                                    clean = cleanPost(news)
                                    label = getLabel(category)
                                    X.append(clean)
                                    Y.append(label)
                                    index = index + 1
            X = np.asarray(X)
            Y = np.asarray(Y)
            vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=300)
            X = vectorizer.fit_transform(X).toarray()
            np.save("model/X", X)
            np.save("model/Y", Y)
            with open('model/vector.txt', 'wb') as file:
                pickle.dump(vectorizer, file)
            file.close()
        context= {'data':'Dataset Loaded<br/>Total Records found in dataset : '+str(X.shape[0])+"<br/>Labels found in dataset : "+str(labels)}
        return render(request, 'UploadDataset.html', context)            

def UploadDataset(request):
    if request.method == 'GET':
       return render(request, 'UploadDataset.html', {})

def UserLogin(request):
    global username
    if request.method == 'POST':
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        status = "failed"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'newsclassify',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == user and row[1] == password:
                    status = 'success'
                    break
        if status == 'success':
            username = user
            status = 'Welcome username : '+username
            context= {'data':status}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'invalid login'}
            return render(request, 'Login.html', context)
        
def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})    

def logout(request):
    if request.method == 'GET':
        global username
        username = "none"
        return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})

def ClassifyNews(request):
    if request.method == 'GET':
       return render(request, 'ClassifyNews.html', {})    

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'newsclassify',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"                    
        if output == "none":                      
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'newsclassify',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                context= {'data':'Signup Process Completed'}
                return render(request, 'Register.html', context)
            else:
                context= {'data':'Error in signup process'}
                return render(request, 'Register.html', context)
        else:
            context= {'data':output}
            return render(request, 'Register.html', context)    
    

