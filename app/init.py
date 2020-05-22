import pandas as pd
from flask import Flask, request, jsonify
from app.analize import Analize
from app.data_loader import DataLoader
from app.preprocessing import Preprocess
data = DataLoader('./forestfires.csv', ';', ',')
file = data.read_file()

x = ['temp', 'RH', 'wind', 'rain']
y = ['fire']

X_train, X_test, y_train, y_test = Preprocess().split(x, y, file)

result = Analize(X_train, y_train).evaluate_model(X_test, y_test)

#input = {'temp': [45.1],
#        'RH': [0.7],
#       'wind': [3.1],
#      'rain': [1.0]}


#input = pd.DataFrame(input, columns = ['temp', 'RH', 'wind', 'rain'])
#print(X_test)
#print(input)


#one_result = Analize(X_train, y_train).testing_model(input)
#one_result2 = Analize(X_train, y_train).testing_model_prob(input)
#print (one_result)
#print (one_result2[:,1])



app = Flask (__name__)

@app.route('/form_example', methods=['POST', 'GET'])
def form():
    if request.method == 'POST':
        temp = request.form.get('temp')
        RH = request.form.get('RH')
        wind = request.form.get('wind')
        rain = request.form.get('rain')
        input = {'temp': temp,
                 'RH': RH,
                 'wind': wind,
                 'rain': rain}
        input = pd.DataFrame(input, columns = ['temp', 'RH', 'wind', 'rain'], index=[0])
        one_result = Analize(X_train, y_train).testing_model_prob(input)
        one_result = one_result[:,1]
        one_result = one_result * 100
        one_result = str(one_result)
        return '<h1> These circumstances have {} % chance to fire '.format(one_result)


    return ''' <form method = "POST">
    Temp <input type="text" name="temp">
    RH <input type="text" name="RH">
    wind <input type="text" name="wind">
    rain <input type="text" name="rain">
    <input type = "submit">
    </form>'''

@app.route('/json_data', methods=['POST'])
def get_json_data():
    req_data = request.get_json()

    temp = req_data['temp']
    RH = req_data['RH']
    wind = req_data['wind']
    rain = req_data['rain']

    input = {'temp': temp,
             'RH': RH,
             'wind': wind,
             'rain': rain}
    input = pd.DataFrame(input, columns=['temp', 'RH', 'wind', 'rain'], index=[0])
    one_result = Analize(X_train, y_train).testing_model_prob(input)
    one_result = one_result[:, 1]
    one_result = one_result * 100
    one_result = float(one_result)
    return jsonify(probability = one_result)

#if __name__ == '__main__':
app.run(debug=False, port=5000)


