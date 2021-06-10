''' 
Description : Sequential RL with shallow PPO algorithm
Based on the original code by Janis Klaise (https://www.janisklaise.com/post/rl-policy-gradients/)
Modified by : Sahil Bhola (June 10th 2021)
'''
import numpy as np 
import gym
from logisticPolicy import logisticPolicy
import matplotlib.pyplot as plt

learningRate = 0.001
discountFactor = 0.99
maxEpisodes = 1000


def runEpisode(env, policyObj):

	accumulatedReward = 0					#reset accumulated rewards
	observation = env.reset()	#reset enviroment to the initial state
	
	stateHistory = []
	actionHistory = []
	rewardHistory = []
	actionProbHistory = []
	done = False

	while not done:
		
		stateHistory.append(observation)	#append state history
		action, actionProb = policyObj.calcAction(observation)	#calc action based on the observed state
		observation, reward, done, info = env.step(action)	#execute action in the enviroment and observe the state {observation} and the obtained reward {reward}
		accumulatedReward += reward
		rewardHistory.append(reward)
		actionHistory.append(action)
		actionProbHistory.append(actionProb)

	return accumulatedReward, np.array(rewardHistory), np.array(stateHistory), np.array(actionHistory), np.array(actionProbHistory)


def trainModel():
	#initialize env
	env = gym.make('CartPole-v0')
	episodeReward = []
	#initialize policy
	policyObj = logisticPolicy(np.random.randn(4), learningRate, discountFactor)
	#executing learning
	for itrain in range(maxEpisodes):
		#execute episode
		accumulatedReward, rewardHistory, stateHistory, actionHistory, actionProbHistory = runEpisode(env, policyObj)

		episodeReward.append(accumulatedReward)

		#update policy
		policyObj.updatePolicy(rewardHistory, stateHistory, actionHistory)
		
		print('episode : {} complete with accumulated reward : {}'.format(itrain, accumulatedReward))
	
	plt.figure
	plt.plot(episodeReward)
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Episode v.s. Reward')
	plt.show()

def main():
	trainModel()

if __name__ == "__main__":
	main()
