import math

def univariateGaussianFunc(Input, mean, stDeviation):
    sigma = stDeviation
    mew = mean

    probability = 1/(sigma*math.sqrt(2*math.pi)) * math.exp(- ((Input-mew)**2)/(2*sigma**2))

    return probability


if __name__ == "__main__":
    ret = univariateGaussianFunc(10, 1, 3)
    print(ret)
