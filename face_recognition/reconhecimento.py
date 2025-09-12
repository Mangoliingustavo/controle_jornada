import cv2
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class CameraApp(App):
    def build(self):
        self.img1 = Image()
        # Abre a câmera 0 (geralmente webcam)
        self.capture = cv2.VideoCapture(0)
        # Atualiza 30 vezes por segundo
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.img1

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # converte BGR para RGB
            buf = cv2.flip(frame, 0)
            buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
            texture = Texture.create(size=(buf.shape[1], buf.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            self.img1.texture = texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    CameraApp().run()