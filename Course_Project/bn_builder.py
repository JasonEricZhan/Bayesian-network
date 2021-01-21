from collections import defaultdict
import numpy as np
from bn import *






class DAG():
      def __init__(self):
          self.G_collection={}
        
        
      def is_directed_acyclic_graph(self,G):
          #dfs
            
          self.color={}
          self.flag=True 
          for i in G:
              if i not in self.color:
                 self.dfs(i,G)

          return self.flag
              
      
      def dfs(self,v,G):
            
          #prevent overwrite  
          if not self.flag:  
             return 
            
          if v not in self.color:
              self.color[v]="RED"

          for adj in G[v]:
              if adj not in self.color and adj in G:
                  self.dfs(adj,G)
              elif adj in self.color and self.color[adj]=='RED':
                  self.flag=False
              
          self.color[v]="BLACK"

      def random_dag(self,nodes, edges):
          
          
          G = defaultdict(list)
          assert (edges<nodes),"cyclic must happened"
             
          while edges > 0:
             a = np.random.randint(0,nodes-1)
             b=a
             while b==a or b in G[a]:
                b = np.random.randint(0,nodes-1)
                #print(nodes)
             G[a].append(b)
             #print(G)
             if self.is_directed_acyclic_graph(G):
                edges -= 1
             else:
                # we closed a loop!
                #print(b)
                G[a].remove(b)
                if not G[a]:
                    del G[a]
          return G 


      def batch_generate(self,nodes_list, edges_list,merge_component):
            
          for i,n_e in enumerate(zip(nodes_list,edges_list)):
              self.G_collection[i]=self.random_dag(n_e[0],n_e[1])
              
            

          New_G=defaultdict(list)
          base=None
          if merge_component:
             for k in self.G_collection.keys():
                 if k>0:
                     nvals=nodes_list[k-1] 
                     if not base:
                        base=nvals
                     else:
                        base+=nvals
                 for v in self.G_collection[k]:
                     if k>0:
                        for i in range(len(self.G_collection[k][v])):
                              self.G_collection[k][v][i]=self.G_collection[k][v][i]+base
                        New_G[v+base]=self.G_collection[k][v]
                     else:
                        New_G[v]=self.G_collection[k][v]
                 
             self.G_collection={}
             self.G_collection[0]=New_G
          
          return self.G_collection

    
    
    
def makefactor(vars,vals):
    phi = discretefactor(set(vars))
    for j,x in enumerate(product(*map((lambda v : [(v,i) for i in range(v.nvals)]),vars))):
        s = {a:b for (a,b) in x}
        phi[s] = vals[j]
    return phi




def isolated_node(G,i):
    if i in G and G[i]:
        return False
    else:
        for v in G.keys():
            #print(v)
            if i in G[v]:
                return False
            
    return True


def builBN(hand_addG,number_of_edge_list,skew,merge_component=True):
    D=DAG()
    
    #Expend to Maximum nodes' number, and will reduce the node with in out degree are zero
    number_of_val_list=[]
    for i in number_of_edge_list:
        number_of_val_list.append(i+1)
    
    G_set=D.batch_generate(number_of_val_list,number_of_edge_list,merge_component)
    
    
    if merge_component:
        number_of_val_list=[sum(number_of_val_list)]
    
    #append to a set of different graph
    ctr=len(G_set)
    for k in hand_addG:
        G_set[ctr]=hand_addG[k][0]
        ctr+=1
    
    bn_dict={}
    ctr=0
    hand_addG_ctr=0
    record_vars=[]
    for G in G_set:
        var_G=defaultdict(list)
        
        adj_G={}
        if G<len(number_of_val_list):
            for num in range(number_of_val_list[ctr]):
                var_i=3 
                adj_G[num]=discretevariable("VAR_"+str(num),var_i)
                if not isolated_node(G_set[G],num):
                    var_G[num].append(adj_G[num])   
        else:
            for num in range(hand_addG[hand_addG_ctr][1]):
                var_i=3 
                adj_G[num]=discretevariable("VAR_"+str(num),var_i)
                if not isolated_node(G_set[G],num):
                    var_G[num].append(adj_G[num]) 
            hand_addG_ctr+=1
                
        record_vars.append(adj_G)
        for i in G_set[G]:
            for v in G_set[G][i]:
                var_G[v].append(adj_G[i])

        
        generatedbn = bn()
        for i in var_G:
            size=var_G[i][0].nvals
            for v in var_G[i][1:]:
                size*=v.nvals
            #skewed
            if skew:  
                alepha=var_G[i][0].nvals*5
                alepha_list=[]
                for j in range(var_G[i][0].nvals):
                    alepha_list.append(alepha)
                    alepha-=5
                    if alepha==0:
                        alepha=1
                CPT=np.random.dirichlet(tuple(alepha_list),size).reshape(-1,)
            else:
                
                CPT=np.random.dirichlet(tuple([2]*var_G[i][0].nvals),size).reshape(-1,)
            if len(var_G[i])>1: #not only one variable combination
                generatedbn.addfactor(makefactor(var_G[i][1:]+[var_G[i][0]],CPT),var_G[i][0])
            else:
                generatedbn.addfactor(makefactor(var_G[i],CPT),var_G[i][0])
            
         
        #test marginalized to 1 or not
        D=None
        for f in generatedbn.factors:
            if not D:
                D=f
            else:
                D*=f
        
        #this threshold is set for some tolerant of precision error
        assert abs(np.round(D.marginalize(D.scope).phi.reshape(-1,1)[0][0],decimals=3)-1)<0.02, "not marginalized to 1"
        bn_dict[G]=generatedbn
        ctr+=1
        
    return bn_dict,record_vars,G_set