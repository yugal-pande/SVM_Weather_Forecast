import pandas as pd
from sklearn.svm import SVC

weather_data = pd.read_csv('weather_data.csv')

X = weather_data.drop('weather_condition', axis=1)
y = weather_data['weather_condition']

feature_names = X.columns

svm_classifier = SVC()

svm_classifier.fit(X, y)

while True:
    precipitation = float(input("Enter precipitation: "))
    humidity = float(input("Enter humidity: "))
    pressure = float(input("Enter pressure (in hPa): "))
    light_intensity = float(input("Enter light intensity: "))
    temperature = float(input("Enter temperature (in Fahrenheit): "))

    input_data = pd.DataFrame([[precipitation, humidity, pressure, light_intensity, temperature,]], columns=feature_names)

    prediction = svm_classifier.predict(input_data)

    print("Predicted weather condition:", prediction[0])
