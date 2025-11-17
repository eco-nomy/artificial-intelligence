from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('models/modelo_xgboost_fraude.pkl', 'rb') as file:
    modelo_fraude = pickle.load(file)

with open('models/label_encoder_type_classes.pkl', 'rb') as file:
    encoder_classes = pickle.load(file)

with open('models/best_threshold.pkl', 'rb') as file:
    best_threshold = pickle.load(file)

with open('models/modelo_xgboost_valor_hora.pkl', 'rb') as file:
    modelo_hora = pickle.load(file)

with open('models/sentence_embedder_valor_hora.pkl', 'rb') as file:
    embedder_hora = pickle.load(file)


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'API do Eco-nomy rodando ✅',
        'modelos_carregados': {
            'fraude': type(modelo_fraude).__name__,
            'valor_hora': type(modelo_hora).__name__
        },
        'versao': '1.0',
        'autores': 'Rafael Kubagawa - Victor Sabelli - Vinicius Soteras'
    })


@app.route('/predict/fraude', methods=['POST'])
def predict_fraude():
    try:
        payload = request.get_json()
        if payload is None:
            return jsonify({'error': 'JSON inválido ou vazio.'}), 400

        required_fields = ['step', 'type', 'amount', 'nameDest']
        missing = [f for f in required_fields if f not in payload]
        if missing:
            return jsonify({'error': f'Campos ausentes: {", ".join(missing)}'}), 400

        import pandas as pd
        data = pd.DataFrame([payload])
        type_val = data.loc[0, 'type']
        if type_val not in encoder_classes:
            return jsonify({'error': f"Valor de 'type' desconhecido: '{type_val}'. Tipos válidos: {sorted(encoder_classes)}"}), 400
        data['type_encoded'] = data['type'].apply(lambda t: encoder_classes.index(t))
        data['isMerchant'] = data['nameDest'].astype(str).str.startswith('M').astype(int)

        features = ['step', 'type_encoded', 'amount', 'isMerchant']
        X = data[features]

        proba = modelo_fraude.predict_proba(X)[:, 1]
        pred = (proba >= best_threshold).astype(int)

        return jsonify({
            'prediction': [{
                'fraude': int(pred[0]),
                'probabilidade': round(float(proba[0]), 4)
            }]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/valor_hora', methods=['POST'])
def predict_valor_hora():
    try:
        entrada = request.get_json()
        if entrada is None:
            return jsonify({'error': 'JSON inválido ou vazio.'}), 400

        if 'title' not in entrada or 'description' not in entrada:
            return jsonify({'error': "Campos 'title' e 'description' são obrigatórios."}), 400

        texto = str(entrada['title']) + " " + str(entrada['description'])
        texto_embedding = embedder_hora.encode([texto])

        hourly_low = float(entrada.get('hourly_low', 20))
        hourly_high = float(entrada.get('hourly_high', 40))
        budget = float(entrada.get('budget', 0))

        X_num = np.array([[hourly_low, hourly_high, budget]])
        X_input = np.hstack([texto_embedding, X_num])

        pred = modelo_hora.predict(X_input)

        return jsonify({'prediction': [{'valor_justo_hora': round(float(pred[0]), 2)}]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(port=5000, debug=True)
