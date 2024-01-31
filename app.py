import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier




app = Flask(__name__)
app.config['upload folder']='uploads'


@app.route('/')
def home():
    return render_template('index.html')
global path

@app.route('/load data',methods=['POST','GET'])
def load_data():
    if request.method == 'POST':

        file = request.files['file']
        filetype = os.path.splitext(file.filename)[1]
        if filetype == '.csv':
            path = os.path.join(app.config['upload folder'], file.filename)
            file.save(path)
            print(path)
            return render_template('load data.html',msg = 'success')
        elif filetype != '.csv':
            return render_template('load data.html',msg = 'invalid')
        return render_template('load data.html')
    return render_template('load data.html')


@app.route('/view data',methods = ['POST','GET'])
def view_data():
    file = os.listdir(app.config['upload folder'])
    path = os.path.join(app.config['upload folder'],file[0])

    global df
    df = pd.read_csv(path)



    print(df)
    return render_template('view data.html',col_name =df.columns.values,row_val = list(df.values.tolist()))

@app.route('/model',methods = ['POST','GET'])
def model():
    if request.method == 'POST':
        global acc1,acc2,acc3,scores1,scores2,scores3
        global df,x_train,y_train,x_test,y_test
        filename = os.listdir(app.config['upload folder'])
        path = os.path.join(app.config['upload folder'],filename[0])
        df = pd.read_csv(path)
        global testsize

        testsize =int(request.form['testing'])
        print(testsize)

        global x_train,x_test,y_train,y_test
        testsize = testsize/100

        df.person_emp_length.fillna(value=df.person_emp_length.mode()[0],inplace=True)
        df.loan_int_rate.fillna(value=df.loan_int_rate.mode()[0],inplace=True)

        df.replace({'N':0,'Y':1},inplace=True)

        df.replace({'RENT':1,'MORTGAGE':0,'OTHER':2,'OWN':3},inplace=True)

        df.replace({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6},inplace=True)

        df.replace({'DEBTCONSOLIDATION':0,'EDUCATION':1,'HOMEIMPROVEMENT':2,'MEDICAL':3,'PERSONAL':4,'VENTURE':5},inplace=True)

        X1 = df.drop(['loan_status'],axis=1)
        Y1 = df['loan_status']       

        from imblearn.over_sampling import SMOTE
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X1, Y1)
        x_train,x_test,y_train,y_test = train_test_split(X_res, y_res,test_size=testsize,random_state=10)
        # print('ddddddcf')
        model = int(request.form['selected'])
        if model == 1:
            x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
            y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

            model = Sequential()
            model.add(Dense(30, activation='relu'))
            model.add(Dense(20, activation='relu'))
            model.add(Dense(1, activation='softmax'))

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=500, epochs=10,validation_data=(x_test, y_test))

            abc=model.predict(x_test)
            scores1 =accuracy_score(abc,y_test)
            acc1 = scores1
            # scores1 = 0.50
            return render_template('model.html',score = round(acc1,4),msg = 'accuracy',selected  = 'ANN')
        elif model == 2:
            rfc = RandomForestClassifier()
            model2 = rfc.fit(x_train,y_train)
            pred2 = model2.predict(x_test)
            scores2 =accuracy_score(y_test,pred2)
            acc2= scores2
            return render_template('model.html',msg = 'accuracy',score = round(acc2,3),selected = 'RANDOM FOREST CLASSIFIER')
        elif model == 3:
            dt = DecisionTreeClassifier()
            model3 = dt.fit(x_train,y_train)
            pred3 = model3.predict(x_test)
            scores3 = accuracy_score(y_test,pred3)
            acc3 = scores3
            return render_template('model.html',msg = 'accuracy',score = round(acc3,3),selected = 'DecisionTreeClassifier')
    


    return render_template('model.html')

@app.route('/prediction',methods = ['POST',"GET"])
def prediction():
    global x_train,x_test,y_train,y_test
    if request.method == 'POST':

        a = request.form['a']
        b = request.form['b']
        c = request.form['c']
        d = request.form['d']
        e = request.form['e']
        f = request.form['f']
        g = request.form['g']
        h = request.form['h']
        i = request.form['i']
        j = request.form['j']
        k = request.form['k']
     

        values = [[a,b,c,d,e,f,g,h,i,j,k]]
        n111 = np.array(values)

        dtc = RandomForestClassifier()
        dtc.fit(x_train,y_train)

        pred = dtc.predict(values)
        print(pred)
        type(pred)

        if pred == [0]:
            msg = "The Predicted Output Is Person Having Loan status  is non default"
        elif pred == [1]:
            msg = "The Predicted Output Is Person Having Lone status is default"

        return render_template('prediction.html',msg =msg)
    return render_template('prediction.html')

@app.route("/graph",methods=['GET','POST'])
def graph():
    scores1 = 0.50
    print(acc2)
    print(acc3)
    i = [scores1,scores2,scores3]
    return render_template('graph.html',i=i)

if __name__ == '__main__':
    app.run(debug=True)