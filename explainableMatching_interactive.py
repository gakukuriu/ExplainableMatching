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

# profile generator from uniform distribution
def profileGenerator():
  rng = np.random.default_rng()
  p = np.zeros((2*n, k), dtype=int)
  for i in range(2*n):
    p[i] = rng.choice(n, k, replace=False)
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
  for i, a in zip(m, range(n)):
    if i != n:
      tl_i = k - (np.where(p[i] == a)[0][0] + 1)
      tl_a = k - (np.where(p[a+n] == i)[0][0] + 1)
      tl_p = tl_p + tl_i + tl_a
  return tl_p



#
# Experiment
#

# times to run the experiment to get an average
repeat = 10
preferenceLengths = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
marketSizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
results = []

def setNumberOfRepeat(n):
  global repeat
  repeat = n

def getNumberOfRepeat():
  global repeat
  return repeat

def experimentalResults():
  global preferenceLengths
  global marketSizes
  global results

  for pl in preferenceLengths:
    averages = []
    setPreferenceLength(pl)
    relativeMarketSizes = marketSizes[marketSizes.index(pl):]
    for mk in relativeMarketSizes:
      setMarketSize(mk)
      average = 0
      for _ in range(repeat):
        average += experiment()
      average = average / repeat
      averages.append(average)
      print('pl:' + str(pl) + ', mk:' + str(mk) + ' is finished!')
    results.append((relativeMarketSizes, averages))
  print('finished!')

def plotResults():
  fig, ax = plt.subplots()
  for i in range(len(results)):
    ax.plot(results[i][0], results[i][1], label='k = ' + str(preferenceLengths[i]))
  ax.set_xlabel('n (market size)')
  ax.set_ylabel('tl(p)')
  ax.legend()