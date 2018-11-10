import time
import datetime

def getTimeStamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def getFileName(K, T, iters, gama, lamb, alpha):
    return "K:"+str(K)+"-T:"+str(T)+"-iters:"+str(iters)+"-gama:"+str(gama)+"-lamb:"+str(lamb)+"-alpha:"+str(alpha)
