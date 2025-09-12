from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.graphics.texture import Texture
import cv2
import requests
import face_recognition
import numpy as np

SERVER_URL = "http://127.0.0.1:5000"  # Endereço do Flask

class CadastroScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None

    def on_pre_enter(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.atualizar_camera, 1.0 / 15.0)

    def on_pre_leave(self):
        if self.capture:
            Clock.unschedule(self.atualizar_camera)
            self.capture.release()

    def atualizar_camera(self, dt):
        if not self.capture:
            return
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.camera_view.texture = texture

    def capturar_embedding_do_frame_atual(self):
        if not self.capture:
            self.ids.status_label.text = "Câmera não inicializada."
            return None
        ret, frame = self.capture.read()
        if not ret:
            self.ids.status_label.text = "Erro ao capturar imagem."
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            self.ids.status_label.text = "Nenhum rosto detectado."
            return None
        if len(face_locations) > 1:
            self.ids.status_label.text = "Apenas um rosto por vez, por favor."
            return None
        return face_recognition.face_encodings(rgb_frame, face_locations)[0]

    def validar_cpf(self, cpf):
        return len(cpf) == 11 and cpf.isdigit()

    def cadastrar_trabalhador(self):
        cpf = self.ids.input_cpf.text.strip()
        nome = self.ids.input_nome.text.strip()
        cargo = self.ids.input_cargo.text.strip()

        if not all([cpf, nome, cargo]):
            self.ids.status_label.text = "Preencha CPF, Nome e Cargo."
            return
        if not self.validar_cpf(cpf):
            self.ids.status_label.text = "CPF inválido (use 11 dígitos)."
            return

        self.ids.status_label.text = "Processando... Olhe para a câmera."
        embedding = self.capturar_embedding_do_frame_atual()
        if embedding is None:
            return

        data = {
            'cpf': cpf,
            'nome': nome,
            'cargo': cargo,
            'embedding': embedding.tolist()
        }

        try:
            response = requests.post(f"{SERVER_URL}/cadastrar", json=data, timeout=10)
            resposta_json = response.json()
            self.ids.status_label.text = resposta_json.get('mensagem', 'Erro no servidor.')
            if response.status_code == 201:
                self.ids.input_cpf.text = ""
                self.ids.input_nome.text = ""
                self.ids.input_cargo.text = ""
        except requests.exceptions.RequestException as e:
            self.ids.status_label.text = "Falha de conexão com o servidor."
            print(f"Erro de request: {e}")

class PontoScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None

    def on_pre_enter(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.atualizar_camera, 1.0 / 15.0)

    def on_pre_leave(self):
        if self.capture:
            Clock.unschedule(self.atualizar_camera)
            self.capture.release()

    def atualizar_camera(self, dt):
        if not self.capture:
            return
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.camera_view.texture = texture

    def capturar_embedding_do_frame_atual(self):
        if not self.capture:
            self.ids.status_label.text = "Câmera não inicializada."
            return None
        ret, frame = self.capture.read()
        if not ret:
            self.ids.status_label.text = "Erro ao capturar imagem."
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            self.ids.status_label.text = "Nenhum rosto detectado."
            return None
        if len(face_locations) > 1:
            self.ids.status_label.text = "Apenas um rosto por vez, por favor."
            return None
        return face_recognition.face_encodings(rgb_frame, face_locations)[0]

    def validar_cpf(self, cpf):
        return len(cpf) == 11 and cpf.isdigit()

    def registrar_ponto(self):
        cpf = self.ids.input_cpf.text.strip()
        if not cpf:
            self.ids.status_label.text = "Digite o CPF para registrar o ponto."
            return
        if not self.validar_cpf(cpf):
            self.ids.status_label.text = "CPF inválido (use 11 dígitos)."
            return

        self.ids.status_label.text = "Processando... Olhe para a câmera."
        embedding = self.capturar_embedding_do_frame_atual()
        if embedding is None:
            return

        data = {
            'cpf': cpf,
            'embedding': embedding.tolist()
        }

        try:
            response = requests.post(f"{SERVER_URL}/registrar_ponto", json=data, timeout=10)
            resposta_json = response.json()
            self.ids.status_label.text = resposta_json.get('mensagem', 'Erro no servidor.')
        except requests.exceptions.RequestException as e:
            self.ids.status_label.text = "Falha de conexão com o servidor."
            print(f"Erro de request: {e}")

class PontoApp(App):
    def build(self):
        return Builder.load_file("meu.kv")

    def on_stop(self):
        for screen in self.root.screens:
            if hasattr(screen, 'capture') and screen.capture:
                screen.capture.release()

if __name__ == "__main__":
    PontoApp().run()
