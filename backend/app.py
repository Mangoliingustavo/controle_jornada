import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import face_recognition
import pickle  # Adicionado para serialização
from pytz import timezone


# --- NOVA ADIÇÃO 2: Definir o fuso horário de Brasília ---
BR_TIMEZONE = timezone('America/Sao_Paulo')

# Configuração do Flask e banco de dados
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# --- NOVA ADIÇÃO 3: Criar um filtro para o Jinja2 (templates) ---
@app.template_filter('to_br_time')
def to_br_time_filter(dt_utc):
    """Converte uma data/hora UTC para o fuso horário de Brasília."""
    if dt_utc is None:
        return "N/A"
    # 1. Informa que a data/hora do banco é UTC
    utc_dt = dt_utc.replace(tzinfo=timezone('UTC'))
    # 2. Converte para o fuso horário de Brasília
    br_dt = utc_dt.astimezone(BR_TIMEZONE)
    return br_dt


# Modelo do banco: Trabalhador
class Trabalhador(db.Model):
    __tablename__ = 'trabalhador'
    id = db.Column(db.Integer, primary_key=True)
    cpf = db.Column(db.String(11), unique=True, nullable=False)
    nome = db.Column(db.String(100), nullable=False)
    cargo = db.Column(db.String(100), nullable=False)
    # Alterado para LargeBinary para melhor performance
    embedding = db.Column(db.LargeBinary, nullable=False)

    def __repr__(self):
        return f'<Trabalhador {self.nome} - {self.cpf}>'


# Modelo do banco: Ponto (histórico)
class Ponto(db.Model):
    __tablename__ = 'ponto'
    id = db.Column(db.Integer, primary_key=True)
    cpf = db.Column(db.String(11), db.ForeignKey('trabalhador.cpf'), nullable=False)
    data_hora = db.Column(db.DateTime, default=datetime.utcnow)
    # --- ALTERAÇÃO 1: Adicionado campo 'tipo' para controlar Entrada/Saída ---
    tipo = db.Column(db.String(10), nullable=False)  # 'entrada' ou 'saida'

    def __repr__(self):
        return f'<Ponto {self.cpf} - {self.tipo} em {self.data_hora}>'


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
        # --- NOVA VERIFICAÇÃO DE ROSTO DUPLICADO ---

        # 1. Pega o novo embedding da requisição e converte para um array numpy.
        novo_embedding = np.array(data['embedding'], dtype=np.float64)

        # 2. Busca TODOS os trabalhadores existentes no banco.
        trabalhadores_existentes = Trabalhador.query.all()

        if trabalhadores_existentes:
            # 3. Prepara a lista de embeddings conhecidos, carregando-os do banco.
            embeddings_conhecidos = [pickle.loads(trabalhador.embedding) for trabalhador in trabalhadores_existentes]

            # 4. Compara o novo rosto com todos os rostos existentes.
            comparacoes = face_recognition.compare_faces(embeddings_conhecidos, novo_embedding, tolerance=0.6)

            # 5. Se houver qualquer correspondência (True na lista), rejeita o cadastro.
            if True in comparacoes:
                return jsonify({'status': 'erro', 'mensagem': 'Rosto já cadastrado para outro trabalhador.'}), 409

        # --- FIM DA VERIFICAÇÃO ---
        # Se o código chegou até aqui, o rosto é único e podemos prosseguir.

        embedding_blob = pickle.dumps(novo_embedding)

        novo_trabalhador = Trabalhador(
            cpf=cpf,
            nome=data['nome'],
            cargo=data['cargo'],
            embedding=embedding_blob
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
        embedding_cadastrado = pickle.loads(trabalhador.embedding)  # Desserializa
        embedding_atual = np.array(data['embedding'], dtype=np.float64)

        resultado_comparacao = face_recognition.compare_faces(
            [embedding_cadastrado], embedding_atual, tolerance=0.6
        )

        if resultado_comparacao[0]:
            # --- ALTERAÇÃO 2: Lógica para determinar se é Entrada ou Saída ---

            # 1. Busca o último registro de ponto para este CPF
            ultimo_ponto = Ponto.query.filter_by(cpf=cpf).order_by(Ponto.data_hora.desc()).first()

            # 2. Decide o tipo do novo registro
            if ultimo_ponto and ultimo_ponto.tipo == 'entrada':
                tipo_atual = 'saida'
            else:
                # Se não houver registro anterior ou se o último foi uma saída, o novo é uma entrada
                tipo_atual = 'entrada'

            # 3. Cria o novo registro com o tipo e a data/hora atuais
            ponto = Ponto(cpf=cpf, tipo=tipo_atual)
            db.session.add(ponto)
            db.session.commit()

            # --- ALTERAÇÃO 3: Mensagem de sucesso personalizada ---
            mensagem = f'{tipo_atual.capitalize()} registrada para {trabalhador.nome}!'
            return jsonify({'status': 'sucesso', 'mensagem': mensagem}), 200
        else:
            return jsonify({'status': 'erro', 'mensagem': 'Rosto não reconhecido'}), 401

    except Exception as e:
        db.session.rollback()  # Garante que a sessão seja revertida em caso de erro
        print(f"Erro ao processar ponto: {e}")
        return jsonify({'status': 'erro', 'mensagem': 'Erro ao processar reconhecimento facial'}), 500


@app.route('/relatorio')
def relatorio_pontos():
    """
    Busca todos os registros de ponto e os exibe em uma página web.
    O filtro 'to_br_time' será aplicado no template HTML.
    """
    try:
        registros = db.session.query(
            Ponto, Trabalhador.nome
        ).join(
            Trabalhador, Ponto.cpf == Trabalhador.cpf
        ).order_by(
            Ponto.data_hora.desc()
        ).all()

        return render_template('relatorio.html', registros=registros)

    except Exception as e:
        print(f"Erro ao gerar relatório: {e}")
        return "<h1>Erro interno ao gerar o relatório</h1>", 500


# Rota para listar trabalhadores
@app.route('/trabalhadores')
def listar_trabalhadores():
    """
    Busca todos os trabalhadores cadastrados e os exibe em uma página web.
    """
    try:
        # Busca todos os registros da tabela Trabalhador, ordenados por nome
        todos_trabalhadores = Trabalhador.query.order_by(Trabalhador.nome).all()

        # Renderiza o template 'trabalhadores.html', passando a lista para ele
        return render_template('trabalhadores.html', trabalhadores=todos_trabalhadores)

    except Exception as e:
        print(f"Erro ao listar trabalhadores: {e}")
        return "<h1>Erro interno ao buscar a lista de trabalhadores</h1>", 500


@app.route('/trabalhador/editar/<int:trabalhador_id>', methods=['POST'])
def editar_trabalhador(trabalhador_id):
    """
    Recebe os dados do formulário do modal e atualiza o trabalhador no banco.
    """
    try:
        # Busca o trabalhador pelo ID fornecido na URL
        trabalhador = Trabalhador.query.get_or_404(trabalhador_id)

        # Pega os novos dados do formulário
        novo_nome = request.form.get('nome')
        novo_cargo = request.form.get('cargo')

        if novo_nome and novo_cargo:
            trabalhador.nome = novo_nome
            trabalhador.cargo = novo_cargo
            db.session.commit()

        # Redireciona de volta para a lista de trabalhadores após a edição
        return redirect(url_for('listar_trabalhadores'))

    except Exception as e:
        db.session.rollback()
        print(f"Erro ao editar trabalhador: {e}")
        return "<h1>Erro interno ao editar o trabalhador</h1>", 500


@app.route('/trabalhador/excluir/<int:trabalhador_id>', methods=['POST'])
def excluir_trabalhador(trabalhador_id):
    """
    Exclui um trabalhador e todos os seus registros de ponto associados.
    """
    try:
        # Busca o trabalhador a ser excluído
        trabalhador_para_excluir = Trabalhador.query.get_or_404(trabalhador_id)

        # IMPORTANTE: Excluir os registros de ponto associados primeiro
        Ponto.query.filter_by(cpf=trabalhador_para_excluir.cpf).delete()

        # Agora exclui o trabalhador
        db.session.delete(trabalhador_para_excluir)
        db.session.commit()

        # Redireciona de volta para a lista de trabalhadores
        return redirect(url_for('listar_trabalhadores'))

    except Exception as e:
        db.session.rollback()
        print(f"Erro ao excluir trabalhador: {e}")
        return "<h1>Erro interno ao excluir o trabalhador</h1>", 500


# Inicialização
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Cria as tabelas (e a nova coluna 'tipo')
    app.run(host='0.0.0.0', port=5000, debug=True)
