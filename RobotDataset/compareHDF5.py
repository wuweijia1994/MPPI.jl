import h5py as h5

f_original = h5.File("half_cheetah_test_original.hdf5", "r+")
f_changed = h5.File("half_cheetah_test_changed.hdf5", "r+")

group_original = f_original["/state"]
group_changed = f_changed["/state"]

for ori, cha in zip(group_original, group_changed):
    print("original", group_original[ori][150, :])
    print("changed", group_changed[cha][150, :])
