from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# === MODELO 1: Classificação de Fraude ===
with open('models/modelo_xgboost_fraude.pkl', 'rb') as file:
    modelo_fraude = pickle.load(file)

with open('models/label_encoder_type_fraude.pkl', 'rb') as file:
    encoder_fraude = pickle.load(file)

# === MODELO 2: Regressão de Preço Justo (Projeto Fixo) ===
with open('models/modelo_xgboost_valor_projeto.pkl', 'rb') as file:
    modelo_projeto = pickle.load(file)

with open('models/tfidf_vectorizer_valor_projeto.pkl', 'rb') as file:
    vectorizer_projeto = pickle.load(file)

# === MODELO 3: Regressão de Valor por Hora ===
with open('models/modelo_xgboost_valor_hora.pkl', 'rb') as file:
    modelo_hora = pickle.load(file)

with open('models/tfidf_vectorizer_valor_hora.pkl', 'rb') as file:
    vectorizer_hora = pickle.load(file)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'API do ECO Nomy rodando ✅',
        'modelos_carregados': {
            'fraude': type(modelo_fraude).__name__,
            'valor_projeto': type(modelo_projeto).__name__,
            'valor_hora': type(modelo_hora).__name__
        },
        'versao': '1.0',
        'autores': 'Rafael Kubagawa - Victor Sabelli - Vinicius Soteras'
    })

@app.route('/predict/fraude', methods=['POST'])
def predict_fraude():
    try:
        data = pd.DataFrame([request.get_json()])
        data['type_encoded'] = encoder_fraude.transform(data['type'])
        data['deltaOrig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
        data['deltaDest'] = data['newbalanceDest'] - data['oldbalanceDest']
        data['isMerchant'] = data['nameDest'].str.startswith('M').astype(int)

        features = ['step', 'type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'oldbalanceDest', 'newbalanceDest', 'deltaOrig', 'deltaDest', 'isMerchant']
        X = data[features]

        pred = modelo_fraude.predict(X)
        proba = modelo_fraude.predict_proba(X)[:, 1]
        return jsonify({'prediction': [{'fraude': int(p), 'probabilidade': round(float(prob), 4)} for p, prob in zip(pred, proba)]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/valor_projeto', methods=['POST'])
def predict_valor_projeto():
    try:
        entrada = request.get_json()
        texto = (entrada['title'] + " " + entrada['description']).lower()
        texto = ''.join(c for c in texto if c.isalnum() or c.isspace())
        texto_vetorizado = vectorizer_projeto.transform([texto])
        pred = modelo_projeto.predict(texto_vetorizado)
        return jsonify({'prediction': [{'valor_justo_projeto': round(float(pred[0]), 2)}]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/valor_hora', methods=['POST'])
def predict_valor_hora():
    try:
        entrada = request.get_json()
        texto = (entrada['title'] + " " + entrada['description']).lower()
        texto = ''.join(c for c in texto if c.isalnum() or c.isspace())
        texto_vetorizado = vectorizer_hora.transform([texto])
        pred = modelo_hora.predict(texto_vetorizado)
        return jsonify({'prediction': [{'valor_justo_hora': round(float(pred[0]), 2)}]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
