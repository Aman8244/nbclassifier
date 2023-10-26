from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the .pkl model
with open('categorical_nb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Perform any necessary data preprocessing
        test = pd.DataFrame([data.animal,data.age,data.temperature,data.Symptom1,data.Symptom2,data.Symptom3],columns=["Animal","Age","Temperature","Symptom 1","Symptom 2","Symptom 3"])
        predictions = model.predict(test)
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
