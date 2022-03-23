from __future__ import print_function
import sys
import yaml
import os
import pickle
import tensorflow as tf
from ttictoc import tic,toc
import threading
import psutil
from datetime import datetime

from sklearn.model_selection import train_test_split
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# def ps_util_monitor(round):
#     global running
#     running = True
#     currentProcess = psutil.Process()
#     cpu_ = []
#     memo_ = []
#     time_ = []
#     report = {}
#     # start loop
#     while running:
#         cpu_percents = currentProcess.cpu_percent(interval=1)
#         mem_percents = currentProcess.memory_percent()
#         ps_time = str(datetime.now())
#         cpu_.append(cpu_percents)
#         memo_.append(mem_percents)
#         time_.append(ps_time)
#
#     report['round'] = round
#     report['cpu'] = cpu_
#     report['memory'] = memo_
#     report['time'] = time_
#
#     with open('/app/resources.txt', '+a') as f:
#         print(report, file=f)
#     # with open('/app/resources.txt', '+a')as fh:
#     #     fh.write(json.dumps(report))


#################################################################################
#################### new way to monitor resourcse

def get_cpu_usage_pct():
    """
    Obtains the system's average CPU load as measured over a period of 500 milliseconds.
    :returns: System CPU load as a percentage.
    :rtype: float
    """
    return psutil.cpu_percent(interval=0.1)

def get_cpu_frequency():
    """
    Obtains the real-time value of the current CPU frequency.
    :returns: Current CPU frequency in MHz.
    :rtype: int
    """
    return int(psutil.cpu_freq().current)

def get_ram_usage():
    """
    Obtains the absolute number of RAM bytes currently in use by the system.
    :returns: System RAM usage in bytes.
    :rtype: int
    """
    return int(psutil.virtual_memory().total - psutil.virtual_memory().available)

def get_ram_total():
    """
    Obtains the total amount of RAM in bytes available to the system.
    :returns: Total system RAM in bytes.
    :rtype: int
    """
    return int(psutil.virtual_memory().total)

def get_ram_usage_pct():
    """
    Obtains the system's current RAM usage.
    :returns: System RAM usage as a percentage.
    :rtype: float
    """
    return psutil.virtual_memory().percent

###########


def ps_util_monitor(round):
    global running
    running = True
    cpu_P = []
    cpu_f = []
    memo_u = []
    memo_T = []
    memo_P = []
    time_ = []
    report = {}
    # start loop
    while running:
        cpu_P.append(get_cpu_usage_pct())
        cpu_f.append(get_cpu_frequency())
        memo_u.append(int(get_ram_usage() / 1024 / 1024))
        memo_T.append(int(get_ram_total() / 1024 / 1024))
        memo_P.append(get_ram_usage_pct())

    report['round'] = round
    report['cpu_p'] = cpu_P
    report['cpu_f'] = cpu_f
    report['memory_u'] = memo_u
    report['memory_t'] = memo_T
    report['memory_p'] = memo_P

    with open('/app/resources.txt', '+a') as f:
        print(report, file=f)
    # with open('/app/resources.txt', '+a')as fh:
    #     fh.write(json.dumps(report))

#################################################################################


def start_monitor(round):
    global t
    # create thread and start it
    t = threading.Thread(target=ps_util_monitor, args=[round])
    t.start()


def stop_monitor():
    global running
    global t
    # use `running` to stop loop in thread so thread will end
    running = False
    # wait for thread's end
    t.join()


def train(model, data, settings):
    global round
    round=1
    print("-- RUNNING TRAINING --", flush=True)

    with open(os.path.join(data, 'trainx.pyp'), 'rb') as fh:
        x_train = pickle.loads(fh.read())
    with open(os.path.join(data, 'trainy.pyp'),'rb') as fh:
        y_train = pickle.loads(fh.read())


    # Xt, X, Yt, Y = train_test_split(x_train, y_train, test_size=settings['testSize'])

    print(" --------------------------------------- ")
    print("x_train shape: : ", x_train.shape)
    print(" --------------------------------------- ")

    # start_monitor(round)
    # tic()

    model.fit(x_train, y_train, batch_size=settings['batch_size'], epochs=settings['epochs'], verbose=True)

    # elapsed = toc()
    #
    # stop_monitor()
    # round += 1
    #
    # with open('/app/time.txt', '+a') as f:
    #     print(elapsed, file=f)



    # print("x_train shape: : ", x_train.shape)
    # model.fit(x_train, y_train, batch_size=settings['batch_size'], epochs=settings['epochs'], verbose=True)

    print("-- TRAINING COMPLETED --", flush=True)
    return model



if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    # with open('../data/layers.yaml', 'r') as fh1:
    #         try:
    #             layers = dict(yaml.safe_load(fh1))
    #         except yaml.YAMLError as e:
    #             raise(e)

    from fedn.utils.kerashelper import KerasHelper
    from models.vgg import create_seed_model

    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])

    model = create_seed_model(dimension=settings['model_dimension'],trainedLayers=settings['trained_Layers']) #,trainL=layers['trainedLayers'])
    model.set_weights(weights)
    import os
    arr = os.listdir('/app/data')
    print("SADI ------------------------ MAIN --------------- SADI")
    print(arr)
    model = train(model, '../data', settings)
    helper.save_model(model.get_weights(), sys.argv[2])


