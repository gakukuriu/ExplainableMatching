import numpy as np
import matplotlib.pyplot as plt


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
def profileGenerator():
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


# agent-proposing deferred acceptance algorithm
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
def experiment():
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


def experiment_average():
  p = profileGenerator()
  _, m = deferredAcceptance(p)
  tl_p = 0
  for i, a in zip(m, range(n)):
    if i != n:
      tl_i = k - (np.where(p[i] == a)[0][0] + 1)
      tl_a = k - (np.where(p[a+n] == i)[0][0] + 1)
      tl_p = tl_p + tl_i + tl_a
  return tl_p / (2*n)

def experiment_1dEuclideanModels():
  p = profileGenerator_1dInterval()
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

def experiment_2dEuclideanModels():
  p = profileGenerator_2dSquare()
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
marketSizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500] #, 600, 700, 800, 900, 1000]
results_IC = []
results_tlAverages_IC = []
results_1D = []
results_tlAverages_1D = []
results_2D = []
results_tlAverages_2D = []


for pl in preferenceLengths:
  averages_IC = []
  tlAverages_IC = []
  averages_1D = []
  tlAverages_1D = []  
  averages_2D = []
  tlAverages_2D = []

  setPreferenceLength(pl)
  relativeMarketSizes = marketSizes[marketSizes.index(pl):]
  for mk in relativeMarketSizes:
    setMarketSize(mk)
    average_IC = 0
    tlAverage_IC = 0
    average_1D = 0
    tlAverage_1D = 0    
    average_2D = 0
    tlAverage_2D = 0
    for _ in range(repeat):
      av_IC, tlav_IC = experiment()
      average_IC += av_IC
      tlAverage_IC += tlav_IC
      av_1D, tlav_1D = experiment_1dEuclideanModels()
      average_1D += av_1D
      tlAverage_1D += tlav_1D      
      av_2D, tlav_2D = experiment_2dEuclideanModels()
      average_2D += av_2D
      tlAverage_2D += tlav_2D
    average_IC = average_IC / repeat
    average_1D = average_1D / repeat
    average_2D = average_2D / repeat

    averages_IC.append(average_IC)
    averages_1D.append(average_1D)
    averages_2D.append(average_2D)

    tlAverage_IC = tlAverage_IC / repeat
#    tlAverages_IC.append(tlAverage)
    print('pl:' + str(pl) + ', mk:' + str(mk) + ' is finished!')
  results_IC.append((relativeMarketSizes, averages_IC))
  results_1D.append((relativeMarketSizes, averages_1D))
  results_2D.append((relativeMarketSizes, averages_2D))

#  results_tlAverages.append((relativeMarketSizes, tlAverages))

#print(results_tlAverages)

fig, ax = plt.subplots()
for i in range(len(results_IC)):
  ax.plot(results_IC[i][0], results_IC[i][1], label='IC Model, k = ' + str(preferenceLengths[i]))
for i in range(len(results_1D)):
  ax.plot(results_1D[i][0], results_1D[i][1], label='Euclidean 1D Model, k = ' + str(preferenceLengths[i]))
for i in range(len(results_2D)):
  ax.plot(results_2D[i][0], results_2D[i][1], label='Euclidean 2D Model, k = ' + str(preferenceLengths[i]))

ax.set_xlabel('n (market size)')
ax.set_ylabel('tl(p)')
ax.legend()
plt.show()

'''
fig, ax = plt.subplots()
for i in range(len(results_tlAverages)):
  ax.plot(results_tlAverages[i][0], results_tlAverages[i][1], label='k = ' + str(preferenceLengths[i]))
ax.set_xlabel('n (market size)')
ax.set_ylabel('average of tl_i(p)/tl_a(p)')
ax.legend()
plt.show()
'''