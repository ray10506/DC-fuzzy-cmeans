from sklearn.datasets import load_wine,load_iris,load_digits
import csv

def read_dataset(name):
    attri = 0
    truth = []
    num = 0
    if name == "wine":
        wine = load_wine()
        dataset = wine['data'].tolist()
        truth = wine['target']
        attri = 3
    elif name == "iris":
        iris = load_iris()
        dataset = iris['data'].tolist()
        truth = iris['target']
        attri = 3
        num = 50
    elif name == "digits":
        digits = load_digits()
        dataset = digits['data'].tolist()
        truth = digits['target']
        attri = 10
    elif name == "haberman":
        path = "data\\haberman.data"
        dataset = open(path , "r")
        dataset = dataset.readlines()
        attri = 2
        for index,i in enumerate(dataset):
            temp = list(map(float,i.split(",")))
            truth.append(int(temp[-1]-1))
            del temp[-1]
            dataset[index] = temp
    elif name == "seeds":
        path = "data\\seeds.txt"
        dataset = open(path , "r")
        dataset = dataset.readlines()
        attri = 3
        for index,i in enumerate(dataset):
            i = i.split("\t")
            n = i.count("")
            for j in range(n):
                del i[i.index("")]
            truth.append(int(i[-1].replace('\n','')))
            del i[-1]
            i = list(map(float,i))
            dataset[index] = i
        num = 70
    elif name == "alcohol":
        path = "data\\AlcoholQCM.txt"
        dataset = open(path , "r")
        dataset = dataset.readlines()
        attri = 5
        for index,i in enumerate(dataset):
            t = i.split("\n")
            t = t[0].split(" ")
            del t[-1]
            t = list(map(float,t))
            dataset[index] = t
            truth.append(index//25)
        num = 25
    elif name == "glass":
        path = "data\\glass.txt"
        dataset = open(path , "r")
        dataset = dataset.readlines()
        for index,i in enumerate(dataset):
            i = i.split(",")
            truth.append(int(i[-1].replace("\n","")))
            del i[-1]
            del i[0]
            i = list(map(float,i))
            dataset[index] = i
        attri = 6
    elif name == "customer":
        dataset = []
        path = "data\\customers.csv"
        attri = 2
        with open(path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0] != 'Channel':
                    dataset.append(list(map(int,row[2:])))
                    truth.append(int(row[0]))
        
    
    return attri, dataset, truth, num