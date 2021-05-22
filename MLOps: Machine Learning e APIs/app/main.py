from flask import Flask, request, jsonify
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./datasets/casas.csv')
colunas = ['tamanho', 'ano', 'garagem']

X = df.drop(['preco'], axis=1)
y = df['preco']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LinearRegression()
_ = modelo.fit(X_train, y_train)


app = Flask(__name__)

@app.route('/')
def home():
    return 'Minha primeira API.'


@app.route('/sentimento/<frase>')
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(to='en')
    polaridade = tb_en.sentiment.polarity
    return 'Polaridade: {}'.format(polaridade)


@app.route('/cotacao/', methods=['POST'])
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])


app.run(debug=True)