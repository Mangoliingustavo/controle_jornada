import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import face_recognition

# Configuração do Flask e banco de dados
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modelo do banco: Trabalhador
class Trabalhador(db.Model):
    __tablename__ = 'trabalhador'
    id = db.Column(db.Integer, primary_key=True)
    cpf = db.Column(db.String(11), unique=True, nullable=False)
    nome = db.Column(db.String(100), nullable=False)
    cargo = db.Column(db.String(100), nullable=False)
    embedding = db.Column(db.String, nullable=False)  # Embedding como JSON string

    def __repr__(self):
        return f'<Trabalhador {self.nome} - {self.cpf}>'

# Modelo do banco: Ponto (histórico)
class Ponto(db.Model):
    __tablename__ = 'ponto'
    id = db.Column(db.Integer, primary_key=True)
    cpf = db.Column(db.String(11), db.ForeignKey('trabalhador.cpf'), nullable=False)
    data_hora = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Ponto {self.cpf} - {self.data_hora}>'

# Função para validar CPF
def validar_cpf(cpf):
    return len(cpf) == 11 and cpf.isdigit()

# Rota inicial
@app.route('/')
def index():
    return "Servidor Flask para Reconhecimento Facial está no ar!"

# Rota para cadastro
@app.route('/cadastrar', methods=['POST'])
def cadastrar_trabalhador():
    data = request.get_json()
    if not all(key in data for key in ['cpf', 'nome', 'cargo', 'embedding']):
        return jsonify({'status': 'erro', 'mensagem': 'Dados incompletos'}), 400

    cpf = data['cpf']
    if not validar_cpf(cpf):
        return jsonify({'status': 'erro', 'mensagem': 'CPF inválido (use 11 dígitos)'}), 400

    if Trabalhador.query.filter_by(cpf=cpf).first():
        return jsonify({'status': 'erro', 'mensagem': 'CPF já cadastrado'}), 409

    try:
        embedding_json = json.dumps(data['embedding'])  # Converte para string
        novo_trabalhador = Trabalhador(
            cpf=cpf,
            nome=data['nome'],
            cargo=data['cargo'],
            embedding=embedding_json
        )
        db.session.add(novo_trabalhador)
        db.session.commit()
        return jsonify({'status': 'sucesso', 'mensagem': 'Trabalhador cadastrado com sucesso'}), 201
    except Exception as e:
        db.session.rollback()
        print(f"Erro ao cadastrar: {e}")
        return jsonify({'status': 'erro', 'mensagem': 'Erro interno no servidor'}), 500

# Rota para registro de ponto
@app.route('/registrar_ponto', methods=['POST'])
def registrar_ponto():
    data = request.get_json()
    if not all(key in data for key in ['cpf', 'embedding']):
        return jsonify({'status': 'erro', 'mensagem': 'Dados incompletos'}), 400

    cpf = data['cpf']
    if not validar_cpf(cpf):
        return jsonify({'status': 'erro', 'mensagem': 'CPF inválido (use 11 dígitos)'}), 400

    trabalhador = Trabalhador.query.filter_by(cpf=cpf).first()
    if not trabalhador:
        return jsonify({'status': 'erro', 'mensagem': 'Trabalhador não encontrado'}), 404

    try:
        embedding_cadastrado = np.array(json.loads(trabalhador.embedding), dtype=np.float64)
        embedding_atual = np.array(data['embedding'], dtype=np.float64)
        resultado_comparacao = face_recognition.compare_faces([embedding_cadastrado], embedding_atual, tolerance=0.6)

        if resultado_comparacao[0]:
            ponto = Ponto(cpf=cpf)
            db.session.add(ponto)
            db.session.commit()
            return jsonify({'status': 'sucesso', 'mensagem': f'Ponto registrado para {trabalhador.nome}!'}), 200
        else:
            return jsonify({'status': 'erro', 'mensagem': 'Rosto não reconhecido'}), 401
    except Exception as e:
        print(f"Erro ao comparar faces: {e}")
        return jsonify({'status': 'erro', 'mensagem': 'Erro ao processar reconhecimento facial'}), 500

# Inicialização
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Cria as tabelas
    app.run(host='0.0.0.0', port=5000, debug=True)
