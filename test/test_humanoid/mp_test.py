import multiprocessing as mp
import random
import string
import time

random.seed(123)

# Define an output queue
output = mp.Queue()
record = mp.Queue()
outside = [1, 2, 3, 4]
# define a example function
def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
                        string.ascii_lowercase
                        + string.ascii_uppercase
                        + string.digits)
                   for i in range(length))

    time.sleep(5-length)
    record.put((1,[length]*5))
    output.put(outside[length])

# Setup a list of processes that we want to run
processes = [mp.Process(target=rand_string, args=(x, output)) for x in range(4)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]
record = [record.get() for p in processes]

print(results)
print(record)
