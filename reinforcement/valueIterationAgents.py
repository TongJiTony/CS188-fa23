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

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # V(s,k+1) = max(sum(prob)[R(s,a,s') + $*V(s',k)])
        # V(s,0) = 0 => initial self.values(counter)
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        for i in range(self.iterations):
            # use batch version of values
            values = self.values.copy()
            for state in mdp.getStates():
                # skip the terminal state
                if mdp.isTerminal(state):
                    continue
                values[state] = self.computeQvalMax(state)
            # update the whole self.values
            self.values = values

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQvalMax(self, state):
        qVals = []
        for action in self.mdp.getPossibleActions(state):
            qVals.append(self.computeQValueFromValues(state, action))
        return max(qVals)

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        total = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            total += prob * (
                self.mdp.getReward(state, action, nextState)
                + self.discount * self.values[nextState]
            )
        return total
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        qVals = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            qVals[action] = self.computeQValueFromValues(state, action)

        return qVals.argMax()
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states.
        preds = dict()
        mdp = self.mdp
        for state in mdp.getStates():
            preds[state] = set()
            preds[state].add(state)

        for state in mdp.getStates():
            for state in mdp.getStates():
                for action in mdp.getPossibleActions(state):
                    for nextState, _prob in mdp.getTransitionStatesAndProbs(
                        state, action
                    ):
                        preds[nextState].update(preds[state])

        # Initialize an empty priority queue.
        priorQueue = util.PriorityQueue()

        # For each non-terminal state s, do: (iterate over states in the order returned by self.mdp.getStates())
        for s in mdp.getStates():
            if mdp.isTerminal(s):
                continue

            # Find the absolute value of the difference between the current value of s in self.values
            # and the highest Q-value across all possible actions from s;
            # call this number diff. Do NOT update self.values[s] in this step.
            diff = abs(self.values[s] - self.computeQvalMax(s))

            # Push s into the priority queue with priority -diff.
            # We use a negative because the priority queue is a min heap,
            # but we want to prioritize updating states that have a higher error.
            priorQueue.push(s, -diff)

        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for i in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if priorQueue.isEmpty():
                break
            # Pop a state s off the priority queue.
            s = priorQueue.pop()
            # Update the value of s (if it is not a terminal state) in self.values.
            if not mdp.isTerminal(s):
                self.values[s] = self.computeQvalMax(s)
            # For each predecessor p of s, do:
            for p in preds[s]:
                # Find the absolute value of the difference between the current value of p in self.values
                # and the highest Q-value across all possible actions from p
                # call this number diff. Do NOT update self.values[p] in this step.
                diff = abs(self.values[p] - self.computeQvalMax(p))
                # If diff > theta, push p into the priority queue with priority -diff
                # as long as it does not already exist in the priority queue with equal or lower priority.
                if diff > self.theta:
                    priorQueue.update(p, -diff)
