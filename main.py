import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

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

@app.route("/", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        provinsi = data.get('provinsi')
        komoditas = data.get('komoditas')
        bulan = data.get('bulan')

        if not provinsi or not komoditas or not bulan:
            return jsonify({"error": "Please provide 'provinsi', 'komoditas', and 'bulan' body"}), 400

        try:
            bulan = int(bulan)
        except ValueError:
            return jsonify({"error": "'bulan' must be an integer"}), 400

        input_data = {
            'Bulan': [bulan]
        }
        input_data.update({f'Provinsi_{provinsi}': [1]})
        input_data.update({f'Komoditas_{komoditas}': [1]})

        input_df = pd.DataFrame(input_data)
        input_df = input_df.reindex(columns=x.columns, fill_value=0)

        print(input_df)

        try:
            prediction = lin_reg.predict(input_df)[0]
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

        return jsonify({
            "provinsi": provinsi,
            "komoditas": komoditas,
            "bulan": bulan,
            "predicted_price": round(prediction)
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=4000, debug=True)
