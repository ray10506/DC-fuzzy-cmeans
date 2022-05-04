

import random 
import numpy as np
from fcmeans import FCM
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn import metrics
from munkres import Munkres
import read_dataset as read
from pyclustering.cluster.fcm import fcm
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


def get_args():
    parser = argparse.ArgumentParser(description='use which dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', metavar='D', type=str, default="iris",
                        help='dataset', dest='dataset')

    return parser.parse_args()




class fuzzy_cmean:
    def __init__(self, k, dataset, num , name, truth):
        self.array = [[] for i in range(len(dataset))]
        self.row = len(dataset)
        self.col = len(dataset[0])
        self.centers = []
        self.labels = []
        self.k = k
        self.truth = []
        self.min = 10000000
        self.max = -1000000
        self.num = num
        self.name = name
        self.GT = truth
    
    def apply_numpy(self, dataset , c_list , missing_index):         
        for c in c_list:
            for i in range(self.row):
                self.array[i].append( dataset[i][c] )
        self.cmean()
        
        
        
    def cmean(self):                                           
        array = np.array(self.array)
        fcm = FCM(n_clusters = self.k ,error = 0.00001)
        fcm.fit(array)
        self.centers = fcm.centers.tolist()
        self.labels = fcm.predict(array).tolist()
      


    def GRA(self, GRS , c , dataset ,complete , miss):
        self.min = 10000000
        self.max = -1000000
        attri = len(GRS[0])
        r_len = len(GRS) - 1
        GRS_total = []
        
        for i in range(r_len):
            for j in range(attri):
                num = abs(GRS[-1][j] - GRS[i][j])
                if num < self.min :
                    self.min = num
                if num > self.max:
                    self.max = num
        
        
        for i in range(r_len):
            gra_sum = 0
            for j in range(attri):
               gra_sum += (self.min + 0.5 * self.max) / (abs(GRS[-1][j] - GRS[i][j]) + 0.5 * self.max )
            
            temp = round(gra_sum / attri , 5)
            GRS_total.append( [ temp , i ])
            
        GRS_total.sort(key = lambda x:x[0])
        GRS_total.reverse()
        
        
        impute = 0
        K = int(round(r_len * 0.5))
        if K ==0:
            K = 1   
        for i in range(K):
            temp = dataset[ GRS[ GRS_total[i][1] ][0] ][c]  
            impute += temp
        impute = round(impute / K , 5)
        return impute
        
    def impute_data_bycluster(self,dataset, c_list, complete, missing_index, update = 0):                          
        sub = []
        for c in c_list:                                        
            missing_temp = missing_index[c].copy()            
            missing_temp2 = missing_index[c].copy()     
            for i in missing_temp:                            
                l = 0
                tar_clus = self.labels[i]                       
                GRS = []
                for index,j in enumerate(self.labels):
                    if j == tar_clus and index not in missing_temp2:       
                        GRS.append([index])
                        for k in complete:                        
                            GRS[l].append(dataset[index][k])
                        l+=1   
                
                if len(GRS) == 0:                              
                    s = 0
                    num = 0
                    for k in range(self.row):
                        num += 1
                        s += dataset[k][c]
                    s = s/num
                    dataset[i][c] = s
                else:
                    GRS.append([i])
                    for k in complete:
                        GRS[l].append(dataset[i][k])
                    value = self.GRA(GRS , c , dataset, complete, missing_temp)
                    if update == 1:
                        sub.append(abs(dataset[i][c] - value))
                    dataset[i][c] = value
                del missing_temp2[missing_temp2.index(i)]
        if update == 1:
            return sub
    


                
    def create_dataset(self,dataset):
        for index,i in enumerate(dataset):
            i = i.split(",")
            del i[-1]
            i = list(map(float,i))
            dataset[index] = i
        self.row = len(dataset)
        self.col = len(dataset[0])
        self.array = [[] for i in range(self.row)]
        return dataset
    
        
        
        
    def accuracy(self, dataset):
        if self.name == "iris" :
            for i in range(self.k):
                count = np.bincount(self.labels[ self.num*i : self.num*(i+1) ])
                num = np.argmax(count)
                if num in self.truth:
                    num_list = []
                    for j in range(self.k):
                        num_list.append(self.labels[ self.num*i : self.num*(i+1) ].count(j))
                    temp = num_list.copy()
                    
                    count = -1000
                    for index,j in enumerate(num_list):
                        if j > count and index not in self.truth:
                            count = j
                            num = index
                temp = [ num for i in range(self.num)]
                self.truth += temp
        elif self.name == "wine":
            l = [0, 59 , 130 , 178 ]         #wine 
            for i in range(self.k): 
                count = np.bincount(self.labels[ l[i] : l[i+1] ])
                num = np.argmax(count)
                if num in self.truth:
                    num_list = []
                    for j in range(self.k):
                        num_list.append(self.labels[ l[i] : l[i+1] ].count(j))
                    temp = num_list.copy()
                    
                    count = -1000
                    for index,j in enumerate(num_list):
                        if j > count and index not in self.truth:
                            count = j
                            num = index
                temp = [ num for i in range(l[i+1] - l[i])]
                self.truth += temp
        else:
            self.truth = self.assignLabel()
        correct = 0
        for index,i in enumerate(self.labels):
            if i == self.truth[index]:
                correct += 1
        acc = correct / self.row   
        f1 = f1_score(self.labels , self.truth, average = 'weighted')
        nmi = metrics.normalized_mutual_info_score(self.labels,self.truth)
        return acc , f1 , nmi
      
        
    def fuzzy(self, array, k):
        initial_centers = kmeans_plusplus_initializer(array, k).initialize()
        
        # create instance of Fuzzy C-Means algorithm
        fcm_instance = fcm(array, initial_centers)
         
        # run cluster analysis and obtain results
        fcm_instance.process()
        clusters = fcm_instance.get_clusters()
        self.centers = fcm_instance.get_centers()                                                          
        membership = fcm_instance.get_membership()                                   
        
        self.labels = [0 for i in range(row)]
        for index,i in enumerate(clusters):
            for j in i:
                self.labels[j] = index
        return self.labels , membership , self.centers
    
    
    def assignLabel(self):
        size = len(self.GT)
        
        costMatrixSize = max(np.amax(self.GT), np.amax(self.labels))+1
        maxCostMatrix = np.zeros((costMatrixSize, costMatrixSize))
        
        for i in range(size):
            maxCostMatrix[self.GT[i]][self.labels[i]] += 1
        
        minCostMatrix = np.zeros((costMatrixSize, costMatrixSize))
        
        max_aggregate = np.amax(maxCostMatrix)
        
        for row in range(costMatrixSize):
            for col in range(costMatrixSize):
                minCostMatrix[row][col] = max_aggregate - maxCostMatrix[row][col]
        m = Munkres()
        indexs = m.compute(minCostMatrix)
        
        adj_GT = []
        for i in range(size):
            adj_GT.append(indexs[self.GT[i]][1])
        return adj_GT



# Normalize the given data with min-max normalization
def normalize_data(data):
    transformer = MinMaxScaler()
    transformer.fit(data)
    data = transformer.transform(data).tolist()
    return data







args = get_args()
ans = []
f1_ans = []
nmi_ans = []
mis_rate = [0.05,0.1,0.2,0.3,0.4,0.5,0.6]
for miss in mis_rate:
    miss_ratio = miss
    acc_total = 0
    f1_total = 0
    nmi_total = 0
                                                 
    for t in range(50):
        truth = []                                      
        num = 0
        attri, dataset , truth, num = read.read_dataset(args.dataset)
        truth = np.array(truth)
        dataset = normalize_data(dataset)
        fc = fuzzy_cmean(attri, dataset, num , args.dataset, truth)  
        
        
        complete = []
        total_num = len(dataset) * len(dataset[0])
        missing = int(total_num * miss_ratio)      
        miss_rc = []             
        
        row = len(dataset)
        col = len(dataset[0]) 
        
        for i in range(missing):
            r = random.randint(0 , row - 1)
            c = random.randint(0 , col - 1)
            while(dataset[r][c] == -1 or (dataset[r].count(-1) == (col-1)) ):
                r = random.randint(0 , len(dataset) - 1)
                c = random.randint(0 , len(dataset[0]) - 1)
            
            dataset[r][c] = -1
            miss_rc.append([r,c])


        missing_num = [ [0,i] for i in range(col) ]      
        missing_index = [ [] for i in range(col) ]        
        
        for i in range(len(dataset)):
            count = dataset[i].count(-1)                    
            pos = -1
            for j in range(count):
                pos = dataset[i].index(-1,pos+1)
                missing_num[pos][0] += 1
                missing_index[pos].append(i)
        
        missing_num.sort(key = lambda x:x[0])               

    
        num = 0                                            
        for index,i in enumerate(missing_num):
            if num!=0:
                num-=1
                continue
            temp = index
            clus_col = [ i[1] ]                                                                
            
            while( (temp != (col-1))  and (missing_num[temp][0] == missing_num[temp+1][0]) ):   
                num+=1
                temp+=1
                
            
            for j in range(num):
                clus_col.append(missing_num[index+j+1][1])
                
                
            if (index == 0) and i[0] != 0:                                             
                mis_index = []
                for l in range(num+1):
                    s = 0
                    count = 0
                    for k in range(row):
                        if dataset[k][missing_num[index + l][1]] != -1:
                            s  += dataset[k][missing_num[index + l][1]]                 
                            count +=1
                    s /= count
                    for k in missing_index[ missing_num[index + l][1] ]:
                        dataset[k][missing_num[index + l][1]] = round(s,5)               
                        mis_index.append( [ k, missing_num[index + l][1] ] )
                mis_index.sort(key = lambda x:x[0]) 
            else:
                fc.impute_data_bycluster(dataset,clus_col,complete,missing_index)
            fc.apply_numpy(dataset , clus_col , missing_index)              
            
            complete.append(i[1])                                                       
            for j in range(num):
                complete.append(missing_num[index+j+1][1])
            
          
            
        acc, f1 , nmi= fc.accuracy(dataset)
        acc_total += acc
        f1_total += f1
        nmi_total += nmi
        fc = None
    
    acc_total = round(acc_total / 50 , 5 )
    f1_total = round(f1_total / 50 , 5 )
    nmi_total = round(nmi_total / 50 , 5 )
    print("miss ratio" , miss_ratio)
    print(acc_total)
    print(f1_total)
    print(nmi_total)
    ans.append(acc_total)
    f1_ans.append(f1_total)
    nmi_ans.append(nmi_total)