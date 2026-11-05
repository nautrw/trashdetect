from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture  # ty:ignore[unresolved-import]
from kivy.uix.image import Image
from ultralytics import YOLO

model = YOLO("./runs/detect/train-2/weights/best.onnx")


class CameraApp(App):
    def __init__(self):
        super().__init__()
        self.paused = False

    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        self.image = Image()
        self.layout.add_widget(self.image)
        
        self.capture = cv2.VideoCapture(0)

        self.pause_button = Button(text="Pause", font_size=14, size_hint=(1,.25))
        self.pause_button.bind(on_press=self.toggle_pause_camera)
        self.layout.add_widget(self.pause_button)

        Clock.schedule_interval(self.update, 1 / 30)

        return self.layout
    
    def toggle_pause_camera(self, *args, **kwargs):
        self.paused = not self.paused
        return True

    def update(self, dt):
        if not self.paused:
            _, frame = self.capture.read()

            # frame must be converted to RGB because opencv defaults to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction_results = model.predict(frame, conf=0.5, verbose=True) # Iterator[Results | Tensor] | list[Results] | list[Tensor]
            # annotated_frame = prediction_results[0].plot() # ty:ignore[not-subscriptable, unresolved-attribute, unused-ignore-comment]

            annotated_frame = next(iter(prediction_results)).plot()  # ty:ignore[unresolved-attribute]

            # frame.shape = (# rows (y), # columns (x), # color channels (3))
            # size param takes a tuple (columns (x), rows (y))
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="rgb")

            texture.blit_buffer(
                annotated_frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte"
            )

            # texture is upside down for some reason
            texture.flip_vertical()

            self.image.texture = texture

    def on_stop(self):
        self.capture.release()


CameraApp().run()
