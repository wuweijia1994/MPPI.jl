using OpenAIGym
env = GymEnv("CartPole-v0")
for i=1:20
    R, T = Episode(env, RandomPolicy())
    info("Episode $i finished after $T steps. Total reward: $R")
end
