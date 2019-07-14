# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
# ---

import time
import os
import datetime
import imageio

def getTimeStamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def getFileName(K, T, iters, gama, lamb, alpha):
    return "K:"+str(K)+"-T:"+str(T)+"-iters:"+str(iters)+"-gama:"+str(gama)+"-lamb:"+str(lamb)+"-alpha:"+str(alpha)


def save_video(queue, filename, fps):
    if not os.path.isdir(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))

    writer = imageio.get_writer(filename, fps=fps)
    while not queue.empty():
        frame = queue.get()
        writer.append_data(frame)
    writer.close()
