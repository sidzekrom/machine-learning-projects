import math

data = []
f = open('agaricus-lepiota.data.txt')

for row in f:
    line = row.split(",")
    line[len(line)-1] = line[len(line)-1][0]
    data.append(line)

poisondict = [{} for i in range(22)]
edibledict = [{} for i in range(22)]
trainingsize = len(data) / 2
numedible = 0
numpoisonous = 0

for i in range(trainingsize):
    currentpoint = data[i]
    if currentpoint[0] == 'p':
        numpoisonous += 1
        for j in range(23):
            if j > 0:
                if currentpoint[j] in poisondict[j-1]:
                    poisondict[j-1][currentpoint[j]] += 1
                else:
                    poisondict[j-1][currentpoint[j]] = 1
    else:
        numedible += 1
        for j in range(23):
            if j > 0:
                if currentpoint[j] in edibledict[j-1]:
                    edibledict[j-1][currentpoint[j]] += 1
                else:
                    edibledict[j-1][currentpoint[j]] = 1

wrongcounter = 0
ranger = trainingsize

for i in range(ranger):
    currentpoint = data[i + trainingsize]
    accedible = 0
    accpoison = 0
    for j in range(22):
        toAddEdible = edibledict[j][currentpoint[j+1]] if\
                      currentpoint[j+1] in edibledict[j]\
                      else 0
        toAddPoisonous = poisondict[j][currentpoint[j+1]] if\
                      currentpoint[j+1] in poisondict[j]\
                      else 0
        accedible += math.log(float(1+toAddEdible)\
                        / float(1+numedible))
        accpoison += math.log(float(1+toAddPoisonous)\
                        / float(1+numpoisonous))
        if toAddEdible == 0:
            accedible -= 12
        if toAddPoisonous == 0:
            accpoison -= 12
    accpoison += math.log(float(numpoisonous)/float(numpoisonous + numedible))
    accedible += math.log(float(numedible)/float(numedible + numpoisonous))
    probedible = math.exp(accedible)
    probpoison = math.exp(accpoison)
    prediction = 'p' if (probpoison > probedible) else 'e'
    if (prediction != currentpoint[0]):
        wrongcounter += 1
print "error", float(wrongcounter) / float(ranger)
