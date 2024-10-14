# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:42:06 2024

@author: goksu
"""




from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Pickle dosyasından modeli ve sütun bilgilerini yükleme
with open('lightgbm_bayesian_model.pkl', 'rb') as file:
    saved_data = pickle.load(file)
    model = saved_data['model']
    model_columns = saved_data['columns']  # Eğitim sırasında kullanılan sütunlar

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON formatında gelen veriyi al
        data = request.get_json()

        # Veriyi DataFrame formatına çevir
        query_df = pd.DataFrame([data])

        # Kategorik değişkenler için one-hot encoding işlemi (eğitimde olduğu gibi)
        query_df_encoded = pd.get_dummies(query_df, columns=['Makina_Adi', 'BazKumasKodu', 'BASKIBoyamaCinsi'])

        # Eksik sütunları doldurmak için eğitimdeki sütunlar kontrol ediliyor
        missing_cols = [col for col in model_columns if col not in query_df_encoded.columns]
        
        # Eksik sütunları tek seferde ekle
        for col in missing_cols:
            query_df_encoded[col] = 0  # Eksik kategoriler için 0 eklenir

        # Sütunları eğitimdeki gibi sıraya koyma
        query_df_encoded = query_df_encoded.reindex(columns=model_columns, fill_value=0)

        # Model ile tahmin yapma
        prediction = model.predict(query_df_encoded)

        # Tahmin sonucunu JSON formatında döndür
        return jsonify({
            'prediction': prediction[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)
