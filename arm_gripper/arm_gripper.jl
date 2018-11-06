using MuJoCo
act = mj.activate(ENV["MUJOCO_KEY_PATH"])

using Distributions
srand(123)

JOINTSNUM = 9
K = 50
T = 20
alpha = 0.1#TODO
lamb = 0.1
TARGETSTATE = [1, 1 ,1, 0, 0, 0]
gama = 0.5

function cost(state)
  end_pos = state[1]
  end_pos = state[2]
  cost = 0
  for i in 1:ndims(A)
    cost += (end_pos[i]-obj_pos[i])^2
  end
  return cost
end

function simulationInit(path="arm_claw.xml")
  model = mj.loadXML(path)
  data = mj.makeData(model)
  return model, data
end

function weightComputation(S)
  lou = min(S)
  yita = 0
  for i in 1:ndims(S)
    yita += exp.((lou-S[i])/lamb)
  end
  w = []
  for i in 1:ndims(S)
    push!(w, exp.((lou - S[i])/lamb)/yita)
  return w
end

function updateControl(U, base_control, w)
  for i in 1:ndims(U):
    for j in 1:ndims(base_control)
      U[i] += base_control[j][i] * w[j]
  return U
end

real_model, real_data = simulationInit()
mu = zeros(JOINTSNUM)
sigma = 5*ones(JOINTSNUM)
distribution = Normal.(mu, sigma)

U = rand.(distribution, T)

for i in 1:100#TODO
  S = zeros(K)
  base_control = []
  temp = []

  for k in 1:K
    sample_data = deepcopy(real_data)
    kexi = rand.(distribution, T)
    push!(base_control, kexi)
    for t in 1:T
      if k < (1-alpha)*K
        v = U[t] + kexi[t]
      else
        v = kexi[t]
      end
      sample_data.ctrl[:] = v
      sample_data.step()


end
