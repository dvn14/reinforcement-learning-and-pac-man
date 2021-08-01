# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()
        for i in range(self.iterations) :
            new_values = util.Counter()
            for state in all_states :
                if not self.mdp.isTerminal(state) :
                    all_actions = self.mdp.getPossibleActions(state)
                    qValues = []
                    for action in all_actions :
                        qValues.append(self.getQValue(state, action))
                    new_values[state] = max(qValues)
            for state in new_values :
                self.values[state] = new_values[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transition_probabilities = self.mdp.getTransitionStatesAndProbs(state, action)
        new_value = 0
        
        for (nextState, prob) in transition_probabilities :
            immediate_reward = self.mdp.getReward(state,action,nextState)
            future_reward = self.getValue(nextState)
            
            new_value += prob*(immediate_reward + self.discount*future_reward)
            
        return new_value
        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state) :
            return None
        else :
            all_actions = self.mdp.getPossibleActions(state)
            qValues = util.Counter()
            for action in all_actions :
                qValues[action] = self.getQValue(state, action)
            
            return qValues.argMax()
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()
        for i in range(self.iterations) :
            state = all_states[i % len(all_states)]
            if not self.mdp.isTerminal(state) :
                all_actions = self.mdp.getPossibleActions(state)
                qValues = []
                for action in all_actions :
                    qValues.append(self.getQValue(state, action))
                self.values[state] = max(qValues)
        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()
        
        "Compute predecessors of all states"
        predecessors = {}
        for state in all_states :
            predecessors[state] = set()
        for state in all_states :
            for action in self.mdp.getPossibleActions(state) :
                for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action) :
                    if prob > 0 :
                        predecessors[nextState].add(state)
        
        "Initialize an empty priority queue"
        myheap = util.PriorityQueue()
        
        "Computing diff"
        for state in all_states :
            if not self.mdp.isTerminal(state) :
                diff = abs(self.getValue(state) - self.computeQvalues(state))
                myheap.push(state, -diff)
                
        "Updating"
        for i in range(self.iterations) :
            if myheap.isEmpty() :
                break
            else :
                state = myheap.pop()
                if not self.mdp.isTerminal(state) :
                    self.values[state] = self.computeQvalues(state)
                for p in predecessors[state] :
                    if self.mdp.isTerminal(p) :
                        diff = abs(self.getValue(p))
                    else :
                        diff = abs(self.getValue(p) - self.computeQvalues(p))
                    
                    if diff > self.theta :
                        myheap.update(p, -diff)

    def computeQvalues(self, state):
        all_actions = self.mdp.getPossibleActions(state)
        qValues = []
        for action in all_actions :
            qValues.append(self.getQValue(state, action))
        
        return max(qValues)
