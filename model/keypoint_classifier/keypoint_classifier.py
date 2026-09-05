import os
import numpy as np
import tensorflow as tf


class KeyPointClassifier:

    def __init__(
        self,
        model_path=None,
        num_threads=1,
    ):

        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        if model_path is None:

            model_path = os.path.join(
                base_dir,
                "keypoint_classifier.keras"
            )

        if not os.path.isfile(model_path):

            raise FileNotFoundError(
                f"Model file not found:\n{model_path}"
            )

        print("Loading model:")
        print(model_path)

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )


    def __call__(self, landmark_list):

        input_data = np.asarray(
            landmark_list,
            dtype=np.float32
        )

        if input_data.ndim == 1:

            input_data = np.expand_dims(
                input_data,
                axis=0
            )

        prediction = self.model.predict(
            input_data,
            verbose=0
        )

        return int(
            np.argmax(
                prediction[0]
            )
        )