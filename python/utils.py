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
