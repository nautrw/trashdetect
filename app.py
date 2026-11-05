from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from datetime import datetime
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
        layout = BoxLayout(orientation='vertical')
        buttons_layout = BoxLayout(orientation='horizontal', size_hint=(1,.25))

        self.image = Image()
        layout.add_widget(self.image)
        
        self.capture = cv2.VideoCapture(0)

        pause_button = Button(text="Pause", font_size=14)
        pause_button.bind(on_press=self.toggle_pause_camera)
        buttons_layout.add_widget(pause_button)

        save_button = Button(text="Save To File", font_size=14)
        save_button.bind(on_press=self.save_image)
        buttons_layout.add_widget(save_button)

        layout.add_widget(buttons_layout)

        Clock.schedule_interval(self.update, 1 / 30)

        return layout

    def save_image(self, *args, **kwargs):
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.image.export_to_png(f'TrashDetect Output {timestamp_str}.png')
    
    def toggle_pause_camera(self, *args, **kwargs):
        self.paused = not self.paused
        return True

    def update(self, dt):
        if not self.paused:
            _, frame = self.capture.read()

            # frame must be converted to RGB because opencv defaults to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # prediction_results: Iterator[Results | Tensor] | list[Results] | list[Tensor]
            prediction_results = model.predict(frame, conf=0.5, verbose=False) 

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
