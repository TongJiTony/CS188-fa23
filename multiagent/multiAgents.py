# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

PACMAN_INDEX = 0
POSITIVE_INFINITE = float("inf")
NEGATIVE_INFINITE = float("-inf")


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        capsuleList = currentGameState.getCapsules()
        foodDistances = []
        capsuleDistances = []
        ghostDistances = []

        if len(foodList) == 0:
            return successorGameState.getScore()

        for food in foodList:
            foodDistances.append(manhattanDistance(newPos, food))
        for capsule in capsuleList:
            capsuleDistances.append(manhattanDistance(newPos, capsule))
        for ghost in newGhostStates:
            ghostPosition = ghost.getPosition()
            ghostDistances.append(manhattanDistance(newPos, ghostPosition))

        # avoid being caught by ghost
        if min(newScaredTimes) == 0 and min(ghostDistances) < 2:
            return -1
        # get capsule to eat the ghost!
        elif len(capsuleDistances) > 0:
            return successorGameState.getScore() - min(capsuleDistances)
        elif min(newScaredTimes) > 0:
            return successorGameState.getScore() - min(ghostDistances)

        return successorGameState.getScore() + 1 / min(foodDistances)


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.getBestAction(gameState)
        util.raiseNotDefined()

    # helper function
    def getBestAction(self, gameState: GameState):
        maxVal = NEGATIVE_INFINITE
        bestAction = Directions.STOP
        actions = gameState.getLegalActions(PACMAN_INDEX)
        for action in actions:
            successor = gameState.generateSuccessor(PACMAN_INDEX, action)
            tempVal = self.value(successor, 0, PACMAN_INDEX + 1)
            if tempVal > maxVal:
                maxVal = tempVal
                bestAction = action
        return bestAction

    def value(self, gameState: GameState, depth: int, agentIndex: int):
        # reach terminate state
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if agentIndex == PACMAN_INDEX:
            return self.maxValue(gameState, depth, agentIndex)
        else:
            return self.minValue(gameState, depth, agentIndex)

    def maxValue(self, gameState: GameState, depth: int, agentIndex: int):
        maxVal = NEGATIVE_INFINITE
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            maxVal = max(maxVal, self.value(successor, depth, agentIndex + 1))
        return maxVal

    def minValue(self, gameState: GameState, depth: int, agentIndex: int):
        minVal = POSITIVE_INFINITE
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            # finish all min layers search in one depth -> back to PACMAN
            if agentIndex + 1 == gameState.getNumAgents():
                minVal = min(minVal, self.value(successor, depth + 1, PACMAN_INDEX))
            else:
                minVal = min(minVal, self.value(successor, depth, agentIndex + 1))
        return minVal


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.getBestAction(gameState)
        util.raiseNotDefined()

    # helper function - run maxValue on the first Pacman layer
    def getBestAction(self, gameState: GameState):
        maxVal = NEGATIVE_INFINITE
        minOption = POSITIVE_INFINITE
        maxOption = NEGATIVE_INFINITE
        bestAction = Directions.STOP
        actions = gameState.getLegalActions(PACMAN_INDEX)
        for action in actions:
            successor = gameState.generateSuccessor(PACMAN_INDEX, action)
            tempVal = self.value(successor, 0, PACMAN_INDEX + 1, minOption, maxOption)
            if tempVal > maxVal:
                maxVal = tempVal
                bestAction = action
            if maxVal > minOption:
                return maxVal
            maxOption = max(maxVal, maxOption)
        return bestAction

    def value(
        self,
        gameState: GameState,
        depth: int,
        agentIndex: int,
        minOption: int,
        maxOption: int,
    ):
        # reach terminate state
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if agentIndex == PACMAN_INDEX:
            return self.maxValue(gameState, depth, agentIndex, minOption, maxOption)
        else:
            return self.minValue(gameState, depth, agentIndex, minOption, maxOption)

    # use maxOption as alpha, minOption as beta to prune;
    # if maxVal > minOption, then the upper min layer would not choose this child tree, prune and return
    # if minVal < maxOption, then the upper max layer would not choose this child tree, prune and return
    # update the maxOption and minOption respectively, but only next child tree would be affected, not parent tree
    def maxValue(
        self,
        gameState: GameState,
        depth: int,
        agentIndex: int,
        minOption: int,
        maxOption: int,
    ):
        maxVal = NEGATIVE_INFINITE
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            maxVal = max(
                maxVal,
                self.value(successor, depth, agentIndex + 1, minOption, maxOption),
            )
            if maxVal > minOption:
                return maxVal
            maxOption = max(maxVal, maxOption)
        return maxVal

    def minValue(
        self,
        gameState: GameState,
        depth: int,
        agentIndex: int,
        minOption: int,
        maxOption: int,
    ):
        minVal = POSITIVE_INFINITE
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            # finish all min layers search in one depth -> back to PACMAN
            if agentIndex + 1 == gameState.getNumAgents():
                minVal = min(
                    minVal,
                    self.value(
                        successor, depth + 1, PACMAN_INDEX, minOption, maxOption
                    ),
                )
                if minVal < maxOption:
                    return minVal
                minOption = min(minVal, minOption)
            else:
                minVal = min(
                    minVal,
                    self.value(successor, depth, agentIndex + 1, minOption, maxOption),
                )
                if minVal < maxOption:
                    return minVal
                minOption = min(minVal, minOption)
        return minVal


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.getBestAction(gameState)
        util.raiseNotDefined()

    # helper function - run maxValue on the first Pacman layer
    def getBestAction(self, gameState: GameState):
        maxVal = NEGATIVE_INFINITE
        bestAction = Directions.STOP
        actions = gameState.getLegalActions(PACMAN_INDEX)
        for action in actions:
            successor = gameState.generateSuccessor(PACMAN_INDEX, action)
            tempVal = self.value(successor, 0, PACMAN_INDEX + 1)
            if tempVal > maxVal:
                maxVal = tempVal
                bestAction = action
        return bestAction

    def value(self, gameState: GameState, depth: int, agentIndex: int):
        # reach terminate state
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if agentIndex == PACMAN_INDEX:
            return self.maxValue(gameState, depth, agentIndex)
        else:
            return self.expectValue(gameState, depth, agentIndex)

    def maxValue(self, gameState: GameState, depth: int, agentIndex: int):
        maxVal = NEGATIVE_INFINITE
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            maxVal = max(maxVal, self.value(successor, depth, agentIndex + 1))
        return maxVal

    # return the expectation value of each goast layer
    def expectValue(self, gameState: GameState, depth: int, agentIndex: int):
        expectVal = 0
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            # finish all ghost layers search in one depth -> back to PACMAN
            if agentIndex + 1 == gameState.getNumAgents():
                expectVal += self.value(successor, depth + 1, PACMAN_INDEX)
            else:
                expectVal += self.value(successor, depth, agentIndex + 1)

        return expectVal / len(actions)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    1. Focus on getting food unless ghost come closer -> use manhatten distance to find minFoodDistance
    2. Get Capsule along the way -> give capsule a proportion of scores
    3. Once eaten capsule, go to kill the ghost first -> if scaredTime, give scaredghost high amount of scores
    """
    "*** YOUR CODE HERE ***"
    finalScore = currentGameState.getScore()
    pacPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    foodDistances = []

    # if the state is already win or lose, just return final score
    if currentGameState.isWin() or currentGameState.isLose():
        return finalScore
    # basic evaluation -> eat the food!
    for food in foodList:
        foodDistances.append(manhattanDistance(pacPos, food))

    finalScore += 1 / min(foodDistances)

    # let's hunt the ghost
    capsuleList = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    capsuleDistances = []
    ghostDistances = []
    for capsule in capsuleList:
        capsuleDistances.append(manhattanDistance(pacPos, capsule))
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDistances.append(manhattanDistance(pacPos, ghostPos))
    ScaredTime = [ghostState.scaredTimer for ghostState in ghostStates]

    # ghost huntable!
    if sum(ScaredTime) > 0:
        # hunt the last shortest ghost
        targetIndex = ScaredTime.index(min(ScaredTime))
        bonus = 100
        finalScore += bonus - ghostDistances[targetIndex]
    # no ghost to hunt, go get the capsule!
    elif len(capsuleDistances) > 0:
        finalScore -= min(capsuleDistances)

    return finalScore
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
