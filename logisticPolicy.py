import numpy as np
import logging

class logisticPolicy():
	def __init__(self, policyParam, learningRate, discountFactor):
		self.theta = policyParam
		self.alpha = learningRate
		self.gamma = discountFactor
		#logging.basicConfig(filename='logFile.log', encoding='utf-8', level=logging.DEBUG)

	def calcAction(self, state):
		actProb = self.calcActionProbability(state) 
		action  = np.random.choice([0, 1], p = actProb)
		
		return action, actProb[action]

	def calcLogGradient(self, state):
		z = np.dot(state, self.theta)
		leftGradient = state*(1-self.calcActivation(z))
		rightGradient = -state*self.calcActivation(z)

		return np.array([leftGradient, rightGradient])

	def calcActionProbability(self, state):
		leftAction = self.calcActivation(np.dot(state, self.theta))
		rightAction = 1-leftAction

		return np.array([leftAction, rightAction])

	def calcActivation(self, z):

		return 1/(1+np.exp(-z))

	def calcAdjustedRewards(self, rewardHistory):
		discounted_rewards = np.zeros(len(rewardHistory))
		cumulative_rewards = 0
		for ii in reversed(range(0, len(rewardHistory))):
			cumulative_rewards = cumulative_rewards * self.gamma + rewardHistory[ii]
			discounted_rewards[ii] = cumulative_rewards

		return discounted_rewards


	def updatePolicy(self, rewardHistory, stateHistory, actionHistory):
		#calculate log gradient of the action prababily given the state,  for the complete observed history (excluding the terminal state))
		grad_log_p = np.array([self.calcLogGradient(state)[action] for state,action in zip(stateHistory, actionHistory)])

		assert grad_log_p.shape == (len(stateHistory), 4)

		#calculate temporarily adjusted rewards
		discountedRewards = self.calcAdjustedRewards(rewardHistory)

		#calculate gradient
		gradient = grad_log_p.T @ discountedRewards

		#parameter update
		self.theta += self.alpha*gradient


		


