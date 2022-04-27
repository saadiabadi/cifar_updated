import sys
import os


import keras
import tensorflow as tf
from models.vgg import create_seed_model

import pickle
import json
import numpy as np
from sklearn import metrics
import yaml
from sklearn.model_selection import train_test_split


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def validate(model, data, settings):
    print("-- RUNNING VALIDATION --", flush=True)


     # Test error (Client has a small dataset set aside for validation)
    # try:
    with open(os.path.join(data, 'testx.pyp'), 'rb') as fh:
        x_test = pickle.loads(fh.read())
    with open(os.path.join(data, 'testy.pyp'), 'rb') as fh:
        y_test = pickle.loads(fh.read())
 print("x_testing shape: : ", x_test.shape)


    try:

        model_score_test = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', model_score_test[0])
        print('Test accuracy:', model_score_test[1])
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        clf_report = metrics.classification_report(y_test.argmax(axis=-1), y_pred)

        print(clf_report)

    except Exception as e:
        print("failed to validate the model {}".format(e), flush=True)
        raise

    report = {
        "classification_report": clf_report,
        "loss": model_score_test[0],
        "accuracy": model_score_test[1],

    }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report


if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise (e)

    from fedn.utils.kerashelper import KerasHelper
    from models.vgg import create_seed_model

    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])

    model = create_seed_model(dimension=settings['model_dimension'])
    model.set_weights(weights)

    report = validate(model, '../data', settings)

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))

