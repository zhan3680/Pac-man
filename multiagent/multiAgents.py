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


import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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

        #Useful information you can extract from a GameState (pacman.py)

        #current GameState
        curFood = currentGameState.getFood()
        curFoodPos = curFood.asList()

        #new GameState
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()

        newFood = successorGameState.getFood()
        newFoodPos = newFood.asList()
        newFoodDis = [manhattanDistance(newPos, food) for food in newFoodPos]

        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPos = [s.getPosition() for s in newGhostStates]
        newGhostDis = [manhattanDistance(newPos, ghost) for ghost in newGhostPos]


        "*** YOUR CODE HERE ***"
        score = 0

        #want to get closer to food or capsules
        if newPos in curFoodPos or newPos in currentGameState.getCapsules():
            score += 10
        else:
            score -= 10*min(newFoodDis)

        #if pacman is invulnerable: (if scaredTime is not the same for each ghost, then check index)
        minGhostDis = min(newGhostDis)
        avgGhostDis = sum(newGhostDis)/len(newGhostDis)
        if max(newScaredTimes) > 0:
            scaredGhostsPos = [ghost.getPosition() for ghost in newGhostStates if ghost.scaredTimer > 0]
            scaredGhostsDis = [manhattanDistance(newPos, ghost) for ghost in scaredGhostsPos]
            minScaredGhostDis = min(scaredGhostsDis)
            if max(newScaredTimes) < minScaredGhostDis:
                score += 200/minScaredGhostDis
        #if the pacman is vulnerable, we want to get further from the ghosts:
        else:
            if minGhostDis != 0 and avgGhostDis!= 0:
                score -= (200/minGhostDis + 200/avgGhostDis)

        return score + (successorGameState.getScore() - currentGameState.getScore())


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth = "2"):
        MultiAgentSearchAgent.__init__(self, evalFn, depth)

    """
    returns the minimax move as well as the score of the pacman at this state if both MIN and MAX play their best moves 
    reference: pseudocode from Game-tree slides p39
    """
    def DFminimax(self, state, player_turn_index):
        best_move = None
        cur_depth = player_turn_index/state.getNumAgents()

        #base case: reach a leave or depth-bound has been reach
        if state.isLose() or state.isWin() or cur_depth == self.depth:
            return best_move, self.evaluationFunction(state)

        #induction step:
        player_index = player_turn_index % state.getNumAgents()
        current_value = 0
        if player_index == 0: #Max node
            current_value -= float('inf')
        else: #Min node
            current_value += float('inf')

        for action in state.getLegalActions(player_index):
            new_state = state.generateSuccessor(player_index, action)
            move, child_value = self.DFminimax(new_state, player_turn_index+1)
            if player_index == 0: #pacman, i.e Max
                if child_value > current_value:
                    current_value = child_value
                    best_move = action
            else: #ghost, i.e Min
                if child_value < current_value:
                    current_value = child_value
                    best_move = action

        return best_move, current_value

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"

        action, utility_for_pacman = self.DFminimax(gameState, 0)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction