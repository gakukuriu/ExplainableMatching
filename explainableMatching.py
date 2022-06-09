import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import normalize_axis_tuple


#
# Parameters
#

# the default value for the market size (total number of agents(objects)) and the length of preference
n = 10
k = 10

# change the market size
def setMarketSize(s):
  global n
  n = s

# change the length of preference
def setPreferenceLength(l):
  global k
  k = l

# get the current market size
def getMarketSize():
  global n
  return n

# get the current length of preference
def getPreferenceLength():
  global k
  return k



#
# Profile generation and DA algorithm, calculation of tl(R)
#

# profile generator based on Impartial Culture Model
def profileGenerator_IC():
  rng = np.random.default_rng()
  p = np.zeros((2*n, k), dtype=int)
  for i in range(2*n):
    p[i] = rng.choice(n, k, replace=False)
  return p

# profile generator based on Euclidean Model
def rankingFromPoints_1dInterval(myPoint, points):
  global n, k
  pointsWithIndexes = list(zip(points, list(range(n))))
  return list(map(lambda x: x[1], sorted(pointsWithIndexes, key=lambda x: abs(myPoint - x[0]))))[:k]

def profileGenerator_1dInterval():
  global n, k
  rng = np.random.default_rng()
  p = np.zeros((2*n, k), dtype=int)

  idealPoints_agents = 2 * rng.random(n) - 1
  idealPoints_objects = 2 * rng.random(n) - 1
  for i in range(n):
    p[i] = rankingFromPoints_1dInterval(idealPoints_agents[i], idealPoints_objects)  
  for i in range(n):
    p[n+i] = rankingFromPoints_1dInterval(idealPoints_objects[i], idealPoints_agents)

  return p

def rankingFromPoints_2dSquare(myPoint, points):
  global n, k
  pointsWithIndexes = list(zip(points, list(range(n))))
  return list(map(lambda x: x[1], sorted(pointsWithIndexes, key=lambda x: np.linalg.norm(myPoint - x[0]))))[:k]  

def profileGenerator_2dSquare():
  global n, k
  rng = np.random.default_rng()
  p = np.zeros((2*n, k), dtype=int)

  idealPoints_agents = 2 * rng.random((n, 2)) - 1
  idealPoints_objects = 2 * rng.random((n, 2)) - 1
  for i in range(n):
    p[i] = rankingFromPoints_2dSquare(idealPoints_agents[i], idealPoints_objects)  
  for i in range(n):
    p[n+i] = rankingFromPoints_2dSquare(idealPoints_objects[i], idealPoints_agents)

  return p

def rankingFromPoints_3dCube(myPoint, points):
  global n, k
  pointsWithIndexes = list(zip(points, list(range(n))))
  return list(map(lambda x: x[1], sorted(pointsWithIndexes, key=lambda x: np.linalg.norm(myPoint - x[0]))))[:k]  

def profileGenerator_3dCube():
  global n, k
  rng = np.random.default_rng()
  p = np.zeros((2*n, k), dtype=int)

  idealPoints_agents = 2 * rng.random((n, 3)) - 1
  idealPoints_objects = 2 * rng.random((n, 3)) - 1
  for i in range(n):
    p[i] = rankingFromPoints_3dCube(idealPoints_agents[i], idealPoints_objects)  
  for i in range(n):
    p[n+i] = rankingFromPoints_3dCube(idealPoints_objects[i], idealPoints_agents)

  return p


# man-proposing deferred acceptance algorithm
def deferredAcceptance(p):
  p_a = np.append(p[:n, :], np.full((n, 1), n), axis=1)
  p_o = np.append(p[n:, :], np.full((n, 1), n), axis=1)
  propose = p_a.copy()[:, :1].T[0]
  accept = p_o.copy()[:, k:].T[0]

  while any(list(map(lambda x: (x != n) and (x != n+1), propose))):
    for p, i in zip(propose, range(n)):
      if (p != n) and (p != n+1):
        p_rank = np.where(p_o[p] == i)[0]
        if p_rank.size > 0:
          a_rank = np.where(p_o[p] == accept[p])[0]
          if p_rank[0] < a_rank[0]:
            if accept[p] != n:
              j = accept[p]
              nextPosition_j = p_a[j][np.where(p_a[j] == p)[0][0] + 1]
              propose[j] = nextPosition_j
            accept[p] = i
            propose[i] = n+1
          else:
            nextPosition = p_a[i][np.where(p_a[i] == p)[0][0] + 1]
            propose[i] = nextPosition
        else:
          nextPosition = p_a[i][np.where(p_a[i] == p)[0][0] + 1]
          propose[i] = nextPosition

  return (propose, accept)


# generate a profile p from the uniform distribution and calculate tl(p)
def subExperimentForDA(profileGenerator):
  p = profileGenerator()
  _, m = deferredAcceptance(p)
  tl_p = 0
  numberOfMatch = 0
  for i, a in zip(m, range(n)):
    if i != n:
      numberOfMatch += 2
      tl_i = k - (np.where(p[i] == a)[0][0] + 1)
      tl_a = k - (np.where(p[a+n] == i)[0][0] + 1)
      tl_p = tl_p + tl_i + tl_a
  return (tl_p, (tl_p / numberOfMatch))



#
# Experiment
#

# times to run the experiment to get an average
repeat = 10

preferenceLengths = [10, 15, 20] #, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
marketSizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def mainExperimentForDA(profileGenerator):
  global n, k, repeat, preferenceLengths, marketSizes
  results = []
  results_tlAverages = []

  for pl in preferenceLengths:
    averages = []
    tlAverages = []
    
    setPreferenceLength(pl)
    relativeMarketSizes = marketSizes[marketSizes.index(pl):]
    for mk in relativeMarketSizes:
      setMarketSize(mk)
      average = 0
      tlAverage = 0
      for _ in range(repeat):
        av, tlAv = subExperimentForDA(profileGenerator)
        average += av
        tlAverage += tlAv
      average = average / repeat
      tlAverage = tlAverage / repeat
      averages.append(average)
      tlAverages.append(tlAverage)
      print('prefelence length:' + str(pl) + ', market size:' + str(mk) + ' is finished!')
    results.append((relativeMarketSizes, averages))
    results_tlAverages.append((relativeMarketSizes, tlAverages))
  return (results, results_tlAverages)


def singlePlot(results, labelForResults, yLabel):
  global preferenceLengths
  fig, ax = plt.subplots()
  for i in range(len(results)):
    ax.plot(results[i][0], results[i][1], label=labelForResults + ', k = ' + str(preferenceLengths[i]))
  ax.set_xlabel('n (market size)')
  ax.set_ylabel(yLabel)
  ax.legend()
  plt.show()


def allPlot(results1, results2, results3, results4, label1, label2, label3, label4, yLabel):
  global preferenceLengths
  fig, ax = plt.subplots()
  for i in range(len(results1)):
    ax.plot(results1[i][0], results1[i][1], label=label1 + ', k = ' + str(preferenceLengths[i]))
  for i in range(len(results2)):
    ax.plot(results2[i][0], results2[i][1], label=label2 + ', k = ' + str(preferenceLengths[i]))
  for i in range(len(results2)):
    ax.plot(results3[i][0], results3[i][1], label=label3 + ', k = ' + str(preferenceLengths[i]))
  for i in range(len(results2)):
    ax.plot(results4[i][0], results4[i][1], label=label4 + ', k = ' + str(preferenceLengths[i]))
  ax.set_xlabel('n (market size)')
  ax.set_ylabel(yLabel)
  ax.legend()
  plt.show()