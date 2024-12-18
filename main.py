import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["POST"])
def predict():
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

        # Ensure 'bulan' is valid
        if bulan not in range(1, 13):
            return jsonify({"error": "'bulan' must be between 1 and 12"}), 400

        # Input data for prediction
        input_data = {
            'Bulan': [bulan]
        }
        input_data.update({f'Provinsi_{provinsi}': [1]})
        input_data.update({f'Komoditas_{komoditas}': [1]})

        input_df = pd.DataFrame(input_data)
        input_df = input_df.reindex(columns=x.columns, fill_value=0)

        # Predict the price for the given month
        try:
            predicted_price = lin_reg.predict(input_df)[0]
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

        # Retrieve the last month's price
        previous_month = bulan - 1
        if previous_month == 0:
            return jsonify({"error": "Cannot calculate percentage change for January (no previous month)"}), 400

        previous_price_row = df[(df['Provinsi'] == provinsi) & 
                                (df['Komoditas'] == komoditas) & 
                                (df['Bulan'] == previous_month)]

        if previous_price_row.empty:
            return jsonify({"error": f"No data available for the previous month ({previous_month})"}), 400

        previous_price = previous_price_row['Harga'].iloc[0]

        # Calculate percentage change
        percentage_change = ((predicted_price - previous_price) / previous_price) * 100

        return jsonify({
            "provinsi": provinsi,
            "komoditas": komoditas,
            "bulan": bulan,
            "predicted_price": round(predicted_price, 2),
            "previous_price": round(previous_price, 2),
            "percentage_change": round(percentage_change, 2)
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=3998, debug=True)
