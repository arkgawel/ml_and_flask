import pandas as pd
from flask import Flask

#app = Flask (__name__)

#@app.route("/")
#def api():
 #   return "Hello flask"


#if __name__ == "name":
#app.run()
from app.analize import Analize
from app.data_loader import DataLoader
from app.preprocessing import Preprocess

data = DataLoader('./forestfires.csv', ';', ',')
file = data.read_file()

x = ['temp', 'RH', 'wind', 'rain']
y = ['fire']

X_train, X_test, y_train, y_test = Preprocess().split(x, y, file)

result = Analize(X_train, y_train).evaluate_model(X_test, y_test)

input = {'temp': [45.1],
         'RH': [0.7],
         'wind': [3.1],
         'rain': [1.0]}


input = pd.DataFrame(input, columns = ['temp', 'RH', 'wind', 'rain'])
print(X_test)
print(input)


one_result = Analize(X_train, y_train).testing_model(input)
one_result2 = Analize(X_train, y_train).testing_model_prob(input)
print (one_result)
print (one_result2[:,1])
