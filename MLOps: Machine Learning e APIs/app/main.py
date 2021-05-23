from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
import pickle

colunas = ['tamanho', 'ano', 'garagem']
modelo = pickle.load(open('./models/modelo.sav', 'rb'))


app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'admin@admin.com'
app.config['BASIC_AUTH_PASSWORD'] = 'admin123'

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return 'Minha primeira API.'


@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(to='en')
    polaridade = tb_en.sentiment.polarity
    return 'Polaridade: {}'.format(polaridade)


@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])


app.run(debug=True)