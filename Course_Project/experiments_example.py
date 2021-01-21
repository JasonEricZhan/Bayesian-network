from bn import *
from bn_builder import *
import matplotlib.pyplot as plt
import itertools  

"""
This is the example, I just change the variablem bnk and queries,skew or not to get the different result

"""
#Symmetric Dirichlet distribution distribution
np.random.seed(2)
g=defaultdict(list)
for i in range(10):
    g[i]=[i+1]
bn_dict,vals_list,G_set=builBN({0:[g,11]},[4,9],skew=False)

bnk=0  # The first graph is the graph with two components



        

bns_set=batch_add_edge(bn_dict) #transfer for bnstructure

node_list=[]
str_node_list=["VAR_6","VAR_9","VAR_10"]


for s in str_node_list:
    for v in bn_dict[bnk].vars:
        if v.name==s:
            node_list.append(v)
            





num_of_e=len(node_list)-1




i_tuple=itertools.product([0,1,2],[0,1,2])




#Likelihood Weighted Sampling
res={}
e_dict={}
for item in i_tuple:
       for i,v in enumerate(node_list[:-1]):
           e_dict[v]=item[i]
       np.random.seed(2)
       sub=[]
       
       for i in range(1000,4500,500):
        
           likelihood_set,likelihood_w=bn_dict[bnk].likelihood_sample(e_dict,i)
           learned_bn=bns_set[bnk].mle_w(likelihood_set,likelihood_w)
           for fs in learned_bn.factors:
             if np.any(np.isnan(fs.phi)):
                fs.phi[np.isnan(fs.phi)]=0
           query=learned_bn.naiveinfval({node_list[-1]},e_dict)
           answer=bn_dict[bnk].naiveinfval({node_list[-1]},e_dict)
           if i not in res:
                  res[i]=abs((query-answer).phi[0])
           else:
                  res[i]+=abs((query-answer).phi[0])

                    
nvals=1
for k in range(len(node_list)):
    nvals*=node_list[k].nvals


for k in res.keys():
    res[k]=res[k]/nvals
    
    
y=[]
x=[]
for k,v in res.items():
    x.append(k)
    y.append(v)                    
                    
                    
plt.plot(x,y)
plt.xlabel('Iteration time')
plt.ylabel('MAE')



#Gibbs Sampling
res={}
for item in i_tuple:
       for i,v in enumerate(node_list[:-1]):
           e_dict[v]=item[i]
       np.random.seed(2)
       for i in range(1000,4000,500):
           gibb_set,gibb_p=bn_dict[bnk].gibb_sample(e_dict,i,int(i*0.5))
           learned_bn=bns_set[bnk].mle(likelihood_set)
           for fs in learned_bn.factors:
             if np.any(np.isnan(fs.phi)):
                fs.phi[np.isnan(fs.phi)]=0
           query=learned_bn.naiveinfval({node_list[-1]},e_dict)
           answer=bn_dict[bnk].naiveinfval({node_list[-1]},e_dict)
           if i not in res:
                  res[i]=abs((query-answer).phi[0])
           else:
                  res[i]+=abs((query-answer).phi[0])



                    
                    
           
nvals=1
for k in range(len(node_list)):
    nvals*=node_list[k].nvals


for k in res.keys():
    res[k]=res[k]/nvals
    
    
y=[]
x=[]
for k,v in res.items():
    x.append(k)
    y.append(v)                    
                    
                    
plt.plot(x,y)
plt.xlabel('Iteration time')
plt.ylabel('MAE')                    
                    
                    
                    

                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
#Skewed distribution                    
np.random.seed(2)
g=defaultdict(list)
for i in range(10):
    g[i]=[i+1]
bn_dict,vals_list,G_set=builBN({0:[g,11]},[4,9],skew=True)

bnk=1


bns_set=batch_add_edge(bn_dict)
res={}



node_list=[]
str_node_list=["VAR_6","VAR_9","VAR_10"]

for s in str_node_list:
    for v in bn_dict[bnk].vars:
        if v.name==s:
            node_list.append(v)
            

num_of_e=len(node_list)-1



i_tuple=itertools.product([0,1,2],[0,1,2])


#Likelihood Weighted Sampling
res={}
e_dict={}
for item in i_tuple:
       for i,v in enumerate(node_list[:-1]):
           e_dict[v]=item[i]
       np.random.seed(2)
       sub=[]
       
       for i in range(1000,4500,500):
        
           likelihood_set,likelihood_w=bn_dict[bnk].likelihood_sample(e_dict,i)
           learned_bn=bns_set[bnk].mle_w(likelihood_set,likelihood_w)
           for fs in learned_bn.factors:
             if np.any(np.isnan(fs.phi)):
                fs.phi[np.isnan(fs.phi)]=0
           query=learned_bn.naiveinfval({node_list[-1]},e_dict)
           answer=bn_dict[bnk].naiveinfval({node_list[-1]},e_dict)
           if i not in res:
                  res[i]=abs((query-answer).phi[0])
           else:
                  res[i]+=abs((query-answer).phi[0])

                    
nvals=1
for k in range(len(node_list)):
    nvals*=node_list[k].nvals


for k in res.keys():
    res[k]=res[k]/nvals
    
    
y=[]
x=[]
for k,v in res.items():
    x.append(k)
    y.append(v)                    
                    
                    
plt.plot(x,y)
plt.xlabel('Iteration time')
plt.ylabel('MAE')



#Gibbs Sampling
res={}
for item in i_tuple:
       for i,v in enumerate(node_list[:-1]):
           e_dict[v]=item[i]
       np.random.seed(2)
       for i in range(500,4000,500):
           gibb_set,gibb_p=bn_dict[bnk].gibb_sample(e_dict,i,int(i*0.5))
           learned_bn=bns_set[bnk].mle(likelihood_set)
           for fs in learned_bn.factors:
             if np.any(np.isnan(fs.phi)):
                fs.phi[np.isnan(fs.phi)]=0
           query=learned_bn.naiveinfval({node_list[-1]},e_dict)
           answer=bn_dict[bnk].naiveinfval({node_list[-1]},e_dict)
           if i not in res:
                  res[i]=abs((query-answer).phi[0])
           else:
                  res[i]+=abs((query-answer).phi[0])



                    
                    
           
nvals=1
for k in range(len(node_list)):
    nvals*=node_list[k].nvals


for k in res.keys():
    res[k]=res[k]/nvals
    
    
y=[]
x=[]
for k,v in res.items():
    x.append(k)
    y.append(v)                    
                    
                    
plt.plot(x,y)
plt.xlabel('Iteration time')
plt.ylabel('MAE')                    
                    