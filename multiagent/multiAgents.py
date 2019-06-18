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
        cur_depth = player_turn_index/state.getNumAgents() + 1
        player_index = player_turn_index % state.getNumAgents()

        #base case: reach a leave or depth-bound has been reached
        if state.isLose() or state.isWin() or (cur_depth > self.depth and player_index == 0):
            return best_move, self.evaluationFunction(state)

        #induction step:
        current_value = 0
        if player_index == 0: #Max node
            current_value -= float('inf')
        else: #Min node
            current_value += float('inf')

        for action in state.getLegalActions(player_index):
            new_state = state.generateSuccessor(player_index, action)
            child_move, child_return = self.DFminimax(new_state, player_turn_index+1)
            if player_index == 0: #pacman, i.e Max
                if child_return > current_value:
                    current_value = child_return
                    best_move = action
            else: #ghost, i.e Min
                if child_return < current_value:
                    current_value = child_return
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth = "2"):
        MultiAgentSearchAgent.__init__(self, evalFn, depth)

    '''
    returns the best move as well as the minimax value with pruning
    '''
    def AlphaBeta(self, state, player_turn_index, alpha, beta):
        best_move = None
        cur_depth = player_turn_index/state.getNumAgents() + 1
        player_index = player_turn_index % state.getNumAgents()

        #base case: reach a leave or depth-bound has been reached
        if state.isLose() or state.isWin() or (cur_depth > self.depth and player_index == 0):
            return best_move, self.evaluationFunction(state)

        #induction step
        for action in state.getLegalActions(player_index):
            new_state = state.generateSuccessor(player_index, action)
            child_move, child_return = self.AlphaBeta(new_state, player_turn_index+1, alpha, beta)
            if player_index == 0: #pacman, i.e Max
                if child_return > alpha:
                    alpha = child_return
                    best_move = action
                    if alpha >= beta:
                        break
            else: #ghost, i.e Min
                if child_return < beta:
                    beta = child_return
                    best_move = action
                    if alpha >= beta:
                        break

        if player_index == 0: #pacman, i.e Max
            return best_move, alpha
        else: #ghost, i.e Min
            return best_move, beta


    def getAction(self, gameState):
        """
          Returns the minimax action based on minimax with alpha-beta pruning
        """
        "*** YOUR CODE HERE ***"
        action, utility_for_pacman = self.AlphaBeta(gameState, 0, -float('inf'), float('inf'))
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth = "2"):
        MultiAgentSearchAgent.__init__(self, evalFn, depth)

    '''
    returns the best move(only for Pacman(Max)) as well as the expectimax value
    '''
    def Expectimax(self, state, player_turn_index):
        best_move = None
        cur_depth = player_turn_index/state.getNumAgents() + 1
        player_index = player_turn_index % state.getNumAgents()

        #base case: reach a leave or depth-bound has been reached
        if state.isLose() or state.isWin() or (cur_depth > self.depth and player_index == 0):
            return best_move, self.evaluationFunction(state)

        #induction step
        count = 0
        current_value = 0
        if player_index == 0: #Max node
            current_value -= float('inf')

        for action in state.getLegalActions(player_index):
            new_state = state.generateSuccessor(player_index, action)
            child_move, child_return = self.Expectimax(new_state, player_turn_index+1)
            if player_index == 0: #pacman, i.e Max
                if child_return > current_value:
                    current_value = child_return
                    best_move = action
            else: #ghost, i.e Min
                count += 1
                current_value += child_return

        if player_index == 0: #pacman, i.e Max
            return best_move, current_value
        else: #ghost, i.e Min
            return best_move, float(current_value)/float(count)


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, utility_for_pacman = self.Expectimax(gameState, 0)
        return action


#calculate average value of a list
def avg(list):
    if len(list) == 0:
        return 0
    else:
        return sum(list)/len(list)

#calculated weighted reciprocal
def weighted_reciprocal(weight, a, b):
    if a != 0 and b != 0:
        return float(weight)*(float(1)/float(a) + float(1)/float(b))
    elif a != 0:
        return float(weight)*(float(1)/float(a))
    elif b != 0:
        return float(weight)*(float(1)/float(b))
    else:
        return 0

#adjusted min that can handle empty lists
def adjusted_min(list):
    if len(list) == 0:
        return 0
    else:
        return min(list)

#return the minimum number of walls the pacman will encounter from coord1 to coord2, following a L-shaped path
def wall_count(walls, pacman, food):
    count1 = 0 #first move horizontally, then move vertically
    count2 = 0 #first move vertically, then move horizontally
    x_step = 0
    y_step = 0
    if food[0] != pacman[0]:
        x_step = (food[0] - pacman[0]) / abs(food[0] - pacman[0])
    if food[1] != pacman[1]:
        y_step = (food[1] - pacman[1])/abs(food[1] - pacman[1])

    if x_step != 0:
        for x in range(pacman[0], food[0], x_step):
            if walls[x][pacman[1]]:
                count1 += 1
            if walls[x][food[1]]:
                count2 += 1
    if y_step != 0:
        for y in range(pacman[1], food[1], y_step):
            if walls[food[0]][y]:
                count1 += 1
            if walls[pacman[0]][y]:
                count2 += 1
    if x_step != 0 and y_step != 0:
        if walls[food[0]][pacman[1]]:
            count1 += 1
        if walls[pacman[0]][food[1]]:
            count2 += 1

    return min(count1, count2)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = 0
    if currentGameState.isWin():
        score += 200000
    if currentGameState.isLose():
        score -= 200000

    pacman_pos = currentGameState.getPacmanPosition()
    food_pos = currentGameState.getFood().asList()
    food_dis = [manhattanDistance(pacman_pos, food) for food in food_pos]
    ghost_pos = currentGameState.getGhostPositions()
    ghost_dis = [manhattanDistance(pacman_pos, ghost) for ghost in ghost_pos]
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghost_state.scaredTimer for ghost_state in ghost_states]
    invulnerable = max(scared_times)

    #first priority: comsume food whenever safe
    num_food_left = currentGameState.getNumFood()
    score -= 500*num_food_left

    #tendency of pacman should be moving towards the nearest food
    walls = currentGameState.getWalls()
    max_manhatten = walls.height + walls.height
    score -= 20 * avg(food_dis)
    score -= (500/max_manhatten)*adjusted_min(food_dis)

    #consider number of walls that lies in the way from pacman to food, if we choose the L-shaped route
    # if closest_food_pos:
    #     num_walls = wall_count(currentGameState.getWalls(), pacman_pos, closest_food_pos)
    #     score -= 20*num_walls

    #if the pacman is vulnerable, try to escape from the ghost, especially when ghost(s) are close
    if invulnerable == 0:
        #take care of ghost
        min_ghost_dis = adjusted_min(ghost_dis)
        avg_ghost_dis = avg(ghost_dis)
        if min_ghost_dis <= 2:
            score -= 1000000

        # only consume a capsule when a ghost is approachable within scared time
        #if min_ghost_dis < (40/2): #SCARED_TIME = 40, PACMAN_SPEED = 1.0, GHOST_SPEED = 1.0/2.0
        capsule_pos = currentGameState.getCapsules()
        capsule_dis = [manhattanDistance(pacman_pos, capsule) for capsule in capsule_pos]
        min_capsule_distance = adjusted_min(capsule_dis)
        score -= 1000*len(capsule_pos)
        #score -= (500/max_manhatten)*min_capsule_distance

    else: #if invulnerable, then chase the closest scared ghost (still need to watch out for ghosts that has been caught
          #once and are thus not scared anymore)
          scared_ghost_pos = [ghost_state.getPosition() for ghost_state in ghost_states if ghost_state.scaredTimer > 0]
          attacking_ghost_pos = [ghost_state.getPosition() for ghost_state in ghost_states if ghost_state.scaredTimer == 0]
          scared_ghost_dis = [manhattanDistance(pacman_pos, ghost) for ghost in scared_ghost_pos]
          attacking_ghost_dis = [manhattanDistance(pacman_pos, ghost) for ghost in attacking_ghost_pos]
          min_scared_ghost_dis = adjusted_min(scared_ghost_dis)
          min_attacking_ghost_dis = adjusted_min(attacking_ghost_dis)
          if min_attacking_ghost_dis <= 2:
              score -= 1000000

          #only chase closest ghost when it is approachable within the scared time rest
          if min_scared_ghost_dis < (invulnerable/3):
              score -= 2000*(min_scared_ghost_dis/max_manhatten)

    # if currentGameState.getNumFood() == 0:
    #     print("in a state where no food is left\n")
    #     if currentGameState.isWin:
    #         print(score)
    #     else:
    #         print("why not winning?\n")
    return score

# Abbreviation
better = betterEvaluationFunction

if __name__ == '__main__':
    for i in range(10,1,-1):
        print(i)