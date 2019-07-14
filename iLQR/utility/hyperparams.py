import numpy as np
from cost import cost_state, cost_action, cost_sum
envs = {
        "humanoid":{
                "path":"/home/wuweijia/GitHub/MPPI/python/humanoid/humanoid.xml",
                "action_weight":np.ones((17,)),
                "converge_threshold":0.05,
                "action_dimension":17,},
                #state_dimension = ,},
        "half_cheetah":{
                "path":"/home/wuweijia/GitHub/MPPI/python/half_cheetah/half_cheetah.xml",
                "action_dimension":6,
                "converge_threshold":0.07,},
        "arm_gripper":{
                "path":"/home/wuweijia/GitHub/MPPI/arm_gripper/arm_claw.xml",
                "converge_threshold":0.07,
                "action_dimension":9,}
        }
