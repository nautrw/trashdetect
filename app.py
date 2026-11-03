import cv2
import os
os.environ["KIVY_WINDOW"] = "sdl2"
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image


class CameraApp(App):
    def build(self):
        self.image = Image()
        self.capture = cv2.VideoCapture(0)

        Clock.schedule_interval(self.update, 1 / 30)

        return self.image

    def update(self, dt):
        ret, frame = self.capture.read()

        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt="rgb"
        )

        texture.blit_buffer(
            frame.tobytes(),
            colorfmt="rgb",
            bufferfmt="ubyte"
        )

        texture.flip_vertical()

        self.image.texture = texture

    def on_stop(self):
        self.capture.release()


CameraApp().run()
