from factorset import factorset
from factor import *
import numpy as np
from itertools import product
import copy
## You need to edit this file to complete the two methods
## that are incomplete below:  bn.sample and bnstructure.mle
## (note: the solutions are only 14 lines of code in total.
##  your solutions may be longer, but if you are writing a lot
##  of code, you are making this too hard on yourself)

## bn is a factorset, augmented to include information about which
## factor corresponds to which CPD
## You need to implement "sample" which is just forward sampling

## bnstructure is just the graph structure for a BN (it has parent
## sets, but no factors)
## You need to implement "mle" which generates a Bayesian network
## (bn) with the same structure, but with the CPTs filled in with their
## maximum likelihood estimates

# for sampling
# use np.random.choice(n,p=[....]) to sample
# from a categorical distribution where [....] is a vector or np.array
# of the probabilities of each outcome and n is the size of this vector/array
#
# Do NOT change the random number seed.  If needed, the testing code
# will set the random number seed before executing "sample"

class NotDAGError:
    pass

# takes in a DAG with interface of "vars" and "parents"
# (both bn and bnstructure below support this interface)
# returns a vector of the nodes in (one possible) topological sort
# (from wikipedia article on topological sorting, but in reverse
#  order so that we only need parents, not children)
def toposort(dag):
    unmarks = set(dag.vars)
    tmarks = set()
    ret = []
    def visit(n):
        if n not in unmarks:
            return
        if n in tmarks:
            raise NotDAGError()
        tmarks.add(n)
        for m in dag.parents(n):
            visit(m)
        tmarks.remove(n)
        unmarks.remove(n)
        ret.append(n)

    while len(unmarks) > 0:
        visit(next(iter(unmarks)))

    return ret


# a Bayesian network is a factorset with
# added information mapping each variable to a factor
# (which is the CPD for that variable)
class bn(factorset):
    # structure should be a bnstructure (see below)
    # factors should be a factorset (see factorset.py)
    def __init__(self):
        super().__init__()
        self._findex = {} # maps variables to their factor index
        self._vindex = [] # maps factor indexes to the variable

    # v should be the variable for which f is the CPD
    def addfactor(self,f,v):
        i = super().addfactor(f)
        self._findex[v] = i
        self._vindex.append(v) #[i] = v
        return i

    def family(self,v):
        return self.factors[self._findex[v]].vars

    def parents(self,v):
        return self.family(v) - {v}

    def __str__(self):
        # needed because str() on a set uses repr(),
        #   not str() on underlying elements
        def settostr(s):
                return '{'+','.join([str(i) for i in s])+'}'

        ret = "variables: %s\n" % settostr(self.vars)
        for v in self.vars:
            ret += "%s: parents = %s\n" % (v,settostr(self.parents(v)))
            ret += str(self.factors[self._findex[v]])
            ret += "\n";
        return ret

    # returns a sample (a dictionary mapping variables to their values,
    # also known as an assignment)
    def forward_sample(self,ignore):
        
        toposort_order=toposort(self)
        root=[]
        for xi in toposort_order:
            if not self.parents(xi):
                root.append(xi)

        dict_path={}
        for xi in toposort_order:
            if xi in ignore:
                dict_path[xi]=ignore[xi]
                continue
            if xi in root:
                i=np.random.choice(xi.nvals,p=self._factors[self._findex[xi]].phi) 
                dict_path[xi]=i
            else:
                dict_temp={}
                for v in self.parents(xi):
                    dict_temp[v]=dict_path[v]
                i=np.random.choice(xi.nvals,p=self._factors[self._findex[xi]][dict_temp]) 
                dict_path[xi]=i
                
        return dict_path
    
    def likelihood_sample_sub(self,evidence,w):
        
        toposort_order=toposort(self)
        root=[]
        for xi in toposort_order:
            if not self.parents(xi):
                root.append(xi)
                
        dict_path={}
        for v in evidence.keys():
            dict_path[v]=evidence[v]
        
        for xi in toposort_order:
            parents_dict={}
            if xi not in evidence.keys():
                if xi not in root:
                    for v in self.parents(xi):
                        parents_dict[v]=dict_path[v]
                    i=np.random.choice(xi.nvals,p=self._factors[self._findex[xi]][parents_dict]) 
                    dict_path[xi]=i
                else:
                    i=np.random.choice(xi.nvals,p=self._factors[self._findex[xi]].phi) 
                    dict_path[xi]=i
            
            else:
                if xi not in root:
                    for v in self.parents(xi):
                        parents_dict[v]=dict_path[v]
                    for v in evidence.keys():
                        if v!=xi:
                           parents_dict[v]=evidence[xi]
                    prob=self._factors[self._findex[xi]][parents_dict][evidence[xi]]
                else:
                    i=evidence[xi]
                    prob=self._factors[self._findex[xi]].phi[i]
                w*=prob
                
        return  dict_path,w        
              
   



    
            
            
    def likelihood_sample(self,evidence,n_iter,return_record=True):
        
        
        #w=1
        sample_list,w_list=[],[]
        for i in range(n_iter):
            
            s,w=self.likelihood_sample_sub(evidence,1)
            w_list.append(w)
            sample_list.append(s)
            
            
        
        result={}
        for sample,w in zip(sample_list,w_list):
            for v in sample:
                if (v,sample[v]) not in result:
                    result[(v,sample[v])]=w
                else:
                    result[(v,sample[v])]+=w
                
        
        
        if not return_record:
            return result
        else:
            return sample_list,w_list
        
        
        
    def gibb_sample_subP(self,x,evidence,previous):    
    
        
        
        
        
        dict_path={}
        p=self._factors[self._findex[x]]
                        
        
        #find childs
        for v in self.vars:
            if v is not x:
                    if x in self.parents(v):
                        """
                        Faster computation from text book
                        We can easily compute the transition model for a 
                        single variable with the probabilit it is related,
                        it is local kernel
                        """
                        p*=self._factors[self._findex[v]]
                        
                            
        
        
        if x in dict_path:
            del dict_path[x]
        
        #set other assignment except xi
        for v in p.vars:
            if v not in dict_path and v!=x:
                if v in evidence:
                    dict_path[v]=evidence[v]
                else:
                    dict_path[v]=previous[v]
           
                    
        p=p.reduce(dict_path)        
        return p/p.marginalize({x})

         
        
        
        
        
        
    def gibb_sample(self,evidence,n_burn,rest,return_record=True):
        
        record_series=[]
        previous=self.forward_sample(evidence)
        
        prob_list=[] #for debug
        
        res=[]
        temp={}
        for i in range(n_burn+rest):
            for v in self.vars:
                if v not in evidence.keys():
                    P_arr=self.gibb_sample_subP(v,evidence,previous)
                    j=np.random.choice(v.nvals,p=P_arr.phi) 
                    prob=P_arr.phi[j]
                    prob_list.append({v.name:{j:prob}})  
                    temp[v]=j
            
            previous=copy.deepcopy(temp)
            if i>=n_burn:
                for v in evidence.keys():
                    previous[v]=evidence[v]
                res.append(previous)
            
        if not return_record:
            return previous
        else:
            return res,prob_list
        
        
    def naiveinfval(self,X,y):
        # in case this helps:
        # y is a dictionary (mapping from variables to values)
        # thus y.keys() is the set of variables
        
        
        join_dst = None
        for i in self.factors:
            if not join_dst :
                join_dst = i.reduce(y)
            else:
                join_dst = join_dst * i.reduce(y)

        return join_dst.marginalize(join_dst.scope-X)/join_dst.marginalize(join_dst.scope)
        
        
        
        
def makefactor(vars,vals):
    phi = discretefactor(set(vars))
    for j,x in enumerate(product(*map((lambda v : [(v,i) for i in range(v.nvals)]),vars))):
        s = {a:b for (a,b) in x}
        phi[s] = vals[j]
    return phi

        
    
def batch_add_edge(bn_set):
    bns_set={}
    for i,bnk in enumerate(bn_set.keys()):
        bns_set[i]=auto_add_edge(bn_set[bnk])
        
    return bns_set





def auto_add_edge(bn):
    
    v_list=[]
    for v in bn.vars:
        v_list.append(v)
        
    bns=bnstructure(set(v_list))
    for v in bn.vars:
        if not bn.parents(v):
            continue
        else:
            for pa in bn.parents(v):
                bns.addedge(pa,v)
                
    return bns


class bnstructure:
    # vars should be a set of variables
    def __init__(self, vars):
        self._par = {v:set() for v in vars}

    def addvar(self,v):
        self._par[v] = set()

    # returns the set of parents of v
    def parents(self, v):
        return self._par[v] if v in self._par else set()

    # returns the set of parents of v
    def family(self, v):
        return self.parents(v).union({v})

    def addedge(self, fromv, tov):
        self._par[tov].add(fromv)

    def deledge(self, fromv, tov):
        self._par[tov].remove(fromv)

    @property
    def vars(self):
        return self._par.keys()

    # returns a bn in which every variable has a uniform
    #   distribution (irrespective of parent values) if v is None,
    #   or all factor values are "v" if v is not None
    # assumes discrete RVs!
    def uniformBN(self,v=None):
        ret = bn()
        for v in self._par:
            if v is None:
                ret.addfactor(discretefactor(self.family(v),1.0/v.nvals),v)
            else:
                ret.addfactor(discretefactor(self.family(v),v),v)
        return ret

    # returns a bn learned using maximum likelihood
    # from the dataset d
    # d is a vector of assignments
    #   (an assignment is a map from variables to values)
    # [this is not the most efficient representation of a dataset,
    #  but it will make the code simple]
    # assumes discrete RVs!
    # assumes complete data!
    def mle(self,d):
        ## to implement
        ret = bn()
        for v in self._par:
            ret.addfactor(discretefactor(self.family(v),0),v)
                
        
        for sample in d:
            for key in sample.keys(): #key is v
                if not self.parents(key):
                    ret._factors[ret._findex[key]][{key:sample[key]}]+=1
                else:
                    dict_temp={}
                    for v in self.family(key):
                        dict_temp[v]=sample[v]
                    ret._factors[ret._findex[key]][dict_temp]+=1
        
        new_ret = bn()
        for v in self._par:
            target_f=ret._factors[ret._findex[v]]
            margin_f=ret._factors[ret._findex[v]].marginalize({v})
            value=target_f/margin_f
            new_ret.addfactor(value,v)
                    
        return new_ret
    
    def mle_w(self,d,w_list):
        ## to implement
        ret = bn()
        for v in self._par:
            ret.addfactor(discretefactor(self.family(v),0),v)
                
        
        for i,sample in enumerate(d):
            for key in sample.keys(): #key is v
                """
                this is to avoid divide error in python, after counting and marginalize will be
                probability
                """
                w=int(np.round(w_list[i],decimals=5)*10**5) 
                if not self.parents(key):
                    ret._factors[ret._findex[key]][{key:sample[key]}]+=w
                else:
                    dict_temp={}
                    for v in self.family(key):
                        dict_temp[v]=sample[v]
                    ret._factors[ret._findex[key]][dict_temp]+=w


        
        new_ret = bn()
        for v in self._par:
            target_f=ret._factors[ret._findex[v]]
            margin_f=ret._factors[ret._findex[v]].marginalize({v})
            value=target_f/margin_f
            new_ret.addfactor(value,v)
                    
        return new_ret
        

