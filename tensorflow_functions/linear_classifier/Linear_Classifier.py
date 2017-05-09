from tensorflow.contrib.learn import LinearClassifier as linear_classifier
from tensorflow.contrib.layers import real_valued_column
import numpy as np


class LinearClassifier:
    def __init__(self,
                 dimension,
                 model_dir=None,
                 n_classes=2,
                 weight_column_name=None,
                 optimizer=None,
                 gradient_clip_norm=None,
                 enable_centered_bias=False,
                 _joint_weight=False,
                 config=None,
                 feature_engineering_fn=None):
        self.feature_columns = [real_valued_column("", dimension=dimension)]

        self.LinearClassifierObject = linear_classifier(feature_columns=self.feature_columns,
                                                        model_dir=model_dir,
                                                        n_classes=n_classes,
                                                        weight_column_name=weight_column_name,
                                                        optimizer=optimizer,
                                                        gradient_clip_norm=gradient_clip_norm,
                                                        enable_centered_bias=enable_centered_bias,
                                                        _joint_weight=_joint_weight,
                                                        config=config,
                                                        feature_engineering_fn=feature_engineering_fn)

    def fit(self, x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None):
        x = np.asarray(x)
        y = np.asarray(y)
        return self.LinearClassifierObject.fit(x=x, y=y, input_fn=input_fn, steps=steps, batch_size=batch_size,
                                               monitors=monitors, max_steps=max_steps)

    def evaluate(self, x=None, y=None, input_fn=None, feed_fn=None, batch_size=None,
                 steps=1, metrics=None, name=None, checkpoint_path=None, hooks=None,
                 log_progress=True):
        x = np.asarray(x)
        y = np.asarray(y)
        return self.LinearClassifierObject.evaluate(x=x, y=y, input_fn=input_fn, feed_fn=feed_fn, batch_size=batch_size,
                                                    steps=steps, metrics=metrics, name=name,
                                                    checkpoint_path=checkpoint_path,
                                                    hooks=hooks, log_progress=log_progress)

    def predict(self, x=None, input_fn=None, batch_size=None, as_iterable=True):
        x = np.asarray(x)
        return list(self.LinearClassifierObject.predict(x=x, input_fn=input_fn, batch_size=batch_size,
                                                   as_iterable=as_iterable))
