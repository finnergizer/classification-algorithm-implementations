import math

def mean(nums):
    return sum(nums)/float(len(nums))

def variance(nums):
    av = mean(nums)
    return sum([pow(x-av, 2) for x in nums])/float(len(nums))

def stdev(nums):
    return math.sqrt(variance(nums))

# Calculate the probability that a value x belongs to a Gaussian distribution with mean and stdev
def calc_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calc_probability_kde(x, means, stdev):
    val = 0
    for ui in means:
        exponent = math.exp(-(math.pow(x-ui,2)/(2*math.pow(stdev,2))))
        val += (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    return val/float(len(means))

def column(matrix, i):
    return [row[i] for row in matrix]
