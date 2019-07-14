import iLQRLR
import h5py


def test_update():
    lr = iLQRLR.DynamicsLRPrior()
    f = h5py.File('train.h5','r+')
    dset = file['/dset']


