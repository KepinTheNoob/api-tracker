import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    df = pd.read_csv('dataSet.csv', usecols=['Provinsi', 'Komoditas', 'Bulan', 'Harga'])
except FileNotFoundError:
    raise FileNotFoundError(f"File {'dataSet.csv'} not found. Ensure the file path is correct.")

month_mapping = {
    'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
    'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
}

df['Bulan'] = df['Bulan'].map(month_mapping)

df_encoded = pd.get_dummies(df, columns=['Provinsi', 'Komoditas'])

x = df_encoded.drop(columns='Harga')
y = df_encoded['Harga']

train_size = int(0.8 * len(x))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

class ManualLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, bulan, harga):
        bulan = np.array(bulan)
        harga = np.array(harga)

        X = np.c_[np.ones(len(bulan)), bulan]
        y = harga

        X_transpose = X.T
        self.coefficients = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
        self.intercept = self.coefficients[0]
        self.slope = self.coefficients[1]

    def predict(self, bulan_prediksi):
        return self.intercept + self.slope * bulan_prediksi


lin_reg = ManualLinearRegression()

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["POST"])

def user_input_prediction():
    try:
        data = request.get_json()

        provinsi = data.get('provinsi')
        komoditas = data.get('komoditas')
        bulan = data.get('bulan')

        if not provinsi or not komoditas or not bulan:
            return jsonify({"error": "Please provide 'provinsi', 'komoditas', and 'bulan' in the body"}), 400

        try:
            bulan = int(bulan)
        except ValueError:
            return jsonify({"error": "'bulan' must be an integer"}), 400

        if bulan not in range(1, 13):
            return jsonify({"error": "'bulan' must be between 1 and 12"}), 400
        
        try:
            bulan = bulan
        except ValueError:
            print("Invalid input. 'Bulan' must be an integer.")
            return

        provinsiLower = provinsi.lower()
        komoditasLower = komoditas.lower()
        filtered_data = df[(df['Provinsi'].str.lower() == provinsiLower) & (df['Komoditas'].str.lower() == komoditasLower)]

        price = np.array(filtered_data['Harga'])
        month = np.array(filtered_data['Bulan'])

        # Train the model using the filtered data
        if(bulan <= filtered_data['Bulan'].iloc[-1]):
                bulan = bulan + 12
        else: 
            bulan = bulan
            print(bulan)
        
        lin_reg.fit(month, price)
        
        try:
            prediction = lin_reg.predict(bulan)
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

        if(bulan > 12):
            bulan = bulan - 12
        else:
            bulan = bulan

        if filtered_data.empty:
            return jsonify({"error": f"No data available for the previous month ({bulan})"}), 400

        previous_price = filtered_data['Harga'].iloc[-1]
        
        # Calculate percentage change
        percentage_change = ((prediction - previous_price) / previous_price) * 100

        return jsonify({
            "provinsi": provinsi,
            "komoditas": komoditas,
            "bulan": bulan,
            "predicted_price": round(prediction),
            "previous_price": round(previous_price),
            "percentage_change": round(percentage_change, 2)
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Run the prediction loop
if __name__ == "__main__":
    app.run(port=3998, debug=True)
