import torch
import MPPI_Mujoco as mppi

import multiprocessing as mp
output = mp.Manager().Queue()

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.linear1(x).clamp(min=0)
        x = self.linear2(x).clamp(min=0)
        y_pred = self.linear3(x)
        return y_pred

def run_mppi(mppi_agent, out):
    mppi_agent.run_MPPI(100)
    out.put(mppi_agent.data)

if __name__ == "__main__":
    import argparse
    import os
    import numpy as np
    import sys
    parser = argparse.ArgumentParser(description='Process which environment to simulate.')
    parser.add_argument('-e', '--env', type=str, nargs='?', default="arm_gripper",
                        help='Enter the name of the environments like: inverted_pendulum, humanoid')
    parser.add_argument('-it', '--iter', type=int, default=100,
                        help='The number of the iterations')

    args = parser.parse_args()
    ENV = args.env
    ITER = args.iter

    hm_env_path = "/home/wuweijia/GitHub/MPPI/python/humanoid/humanoid.xml"
    hc_env_path = "/home/wuweijia/GitHub/MPPI/python/half_cheetah/half_cheetah.xml"
    ag_env_path = "/home/wuweijia/GitHub/MPPI/python/arm_gripper/arm_claw.xml"
    ad_env_path = "/home/wuweijia/GitHub/MPPI/python/arm_door/pr2_arm3d_door.xml"

    #ip_args = {"JOINTSNUM":1, "K":20, "T":500, "alpha":0.1, "lamb":0.05, "gama":0.5, "render":"RECORD", "env_path":ip_env, "mu":None, "sigma":None}

    hm_args = {"JOINTSNUM":17, "K":200, "T":100, "alpha":0.1, "lamb":0.5, "gama":0.5, "render":"RECORD", "env_path":hm_env_path, "mu":np.zeros(17), "sigma":0.05*np.eye(17)}

    ag_args = {"JOINTSNUM":9, "K":50, "T":50, "alpha":0.1, "lamb":0.1, "gama":0.5, "render":"RECORD", "env_path":ag_env_path, "mu":np.zeros(9), "sigma":5*np.eye(9)}

    ad_args = {"JOINTSNUM":6, "K":50, "T":50, "alpha":0.1, "lamb":0.1, "gama":0.5, "render":"RECORD", "env_path":ad_env_path, "mu":np.zeros(6), "sigma":100*np.eye(6)}

    hc_args = {"JOINTSNUM":6, "K":50, "T":50, "alpha":0.1, "lamb":0.1, "gama":0.5, "render":"RECORD", "env_path":hc_env_path, "mu":np.zeros(6), "sigma":5*np.eye(6)}

    args = {"humanoid":hm_args, "robot_arm":ag_args, "half_cheetah":hc_args, "arm_door":ad_args}

    if ENV not in args:
        print("There is no environment: "+ENV)
    else:
        env_arg = args[ENV]

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 25, 14, 50, 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out).to(device)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for t in range(20):
        # Setup a list of processes that we want to run
        import datetime
        starttime = datetime.datetime.now()
        processes = [mp.Process(target=run_mppi, args=(mppi.MPPI(env_arg), output)) for x in range(10)]

        # Run processes
        for p in processes:
                p.start()

        # Exit the completed processes
        for p in processes:
                p.join()

        endtime = datetime.datetime.now()
        print("Time: ", (endtime - starttime).seconds)
        #exit()

        #import pdb; pdb.set_trace()
        data = {}
        # Get process results from the output queue
        results = [output.get() for p in processes]

        data["states"] = np.asarray([results[i]["states"] for i, p in enumerate(processes)])
        data["actions"] = np.asarray([results[i]["actions"] for i, p in enumerate(processes)])
        #mppi_agent = mppi.MPPI(env_arg)
        #mppi_agent.run_MPPI(ITER)

        # Forward pass: Compute predicted y by passing x to the model
        import pdb; pdb.set_trace()
        batchesNumber = ITER//N
        #for _ in range(10):
        for i in range(batchesNumber):
            #import pdb; pdb.set_trace()
            print(i*N, i*N+N)
            x = torch.from_numpy(data["states"][i*N:i*N+N, :]).float().to(device)
            y = torch.from_numpy(data["actions"][i*N:i*N+N, :]).float().to(device)

            y_pred = model(x)

            # Compute and print loss
            loss = criterion(y_pred, y)
            print("iteration: ", t, "batch: ", i, "loss: ", loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), "./mppi_model.mdl")
"""
"""
"""
"""
