#! python 
#! -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


Lp = 5


# Numero clienti n
n = 9
# Capacita mezzo C
C = 150
# Tempo di recupero monete T
T = 15
# Tempi tij
M = np.matrix("""0 39 42 16 52 30 33 49 47 14;
              39 0 42 54 51 71 72 43 27 33;
              42 42 0 30 51 48 52 48 29 32;
              16 54 30 0 16 21 36 53 76 68;
              52 51 51 16 0 21 46 47 44 76;
              30 71 48 21 21 0 60 39 37 29;
              33 72 52 36 46 60 0 36 63 33;
              49 43 48 53 47 39 36 0 49 16;
              47 27 29 76 44 37 63 49 0 39;
              14 33 32 68 76 29 33 16 39 0""")
# quantità da ritirare
q = np.array([0,133,80,40,96,15,37,62,15,100])

# minuti nel giorno
O = 8 * 60

# giorni a settimana
G = 5

# def diff(S1,S2):
#     """Verifica funzione obiettivo"""
#     return True  # return False


# def _permutation_plus(Sin,3):
#     """Restituisce 3 archi (i,j) non-adiacenti (ed eventualmente con condizioni limitative)"""
# #    for ...
# #        yield triplet
#     pass

# def _remove(S,triplet):
#     """Elimina gli archi indicati in triplet da S"""
    
#     return S

# def _readd(S,triplet):
#     """Riaggiunge gli archi che insistono sui nodi indicati da triplet a S tranne proprio triplet"""
    
#     yield S

# def vicinato(S):
#     # ...
#     Sin = S
#     while True:
#         L = len(Sin)
#         for triplet in _permutation_plus(Sin,3):
#             Sp = _remove(Sin,triplet)   # Soluzione iniziale senza la terna degli archi levati
#             for Splus in _readd(Sp,triplet):                                   
#                 yield Splus
        

# def localsearch(iniSol):
#     Sin = iniSol

#     while True:
#         Sols = []
#         for S in vicinato(Sin):
#             if diff(S,Sin)>0:
#                 Sols.append(S)
#                 continue
#         if len(Sols)>0:
#             Sin = maxsol(Sols)
#             continue
    
#     # in Sin c'e la soluzione
#     return Sin

# def constructTour():
#     pass

# def calculateStarts():
    
#     return []

def create_graph(M,first=0):
    (n,m) = M.shape
    print "LEN=",n,m
    _g = nx.Graph()
    _l = first
    for i in range(n):        
        print "ADD NODE",_l
        _g.add_node(_l)
        _l += 1
    for i in range(n):
        for j in range(m):
            if i!=j:
                print "ADD EDGE",i+first,j+first,M[i,j]
                _g.add_edge(i+first,j+first,weight=M[i,j])
    return _g

def find_min(G,N):
    mv = np.inf
    node = 0
#    print G[N].items()
    for n,nbrs in sorted(G[N].items()):
#        print n,nbrs
        data=nbrs['weight']
        if data<mv: 
#            print('(%d -> %d, %.3f)' % (N,n,data))
            mv = data
            node = n
    return (node,mv)

def construct(G,N):
    H = nx.Graph(G)
    Nn = len(G.nodes())

    # trova il primo cliente
    node = N
    tot = 0.0
    Tours = []
    Tour = [ 0 ]
    Q = 0.0
    Day = 0
    Day_T = 0.0
    for i in range(Nn):
        old = node
        node,l = find_min(H,node)
        print node,l,tot,q[node]
        if Q + q[node]>C: # Nuovo Tour
            Tour.append(0)
            print N,old,M[0,old]
            return_time = M[N,old]
            Tours.append([0,tot+return_time,Q,Tour])
            Tour = [0,]
            Q = 0.0
            tot = 0.0
            
        Q += q[node]
        tot += l + 15
        Tour.append(node)
        H.remove_node(old)  
    gg = 0
    tx = 0
    Ts = []
    for g,tt,qq,tour in Tours :
        tx += tt
        if tx > O:
            gg += 1
            tx = tt
        Ts.append([gg,tt,qq,tour])
    print Ts

def main():
    R = None
    multiStartSolutionArray = calculateStarts()
    for S in multiStartSolutionArray:
        S1 = localsearch(S)
        if diff(S1,R)>0:
            R=S1
    print R

if __name__=="__main__":
    G = create_graph(M)  # Grafo completo
    Mminus = M[1:,1:] # Solo i clienti
    print Mminus
    H = create_graph(Mminus,1) # Grafo dei clienti
    Mid = M[0:,0]
    print Mid
    K = create_graph(Mid) # Grafo dal magazzino a ogni cliente
    T=nx.minimum_spanning_tree(H)
    nx.draw(T,pos=nx.spring_layout(H))
    construct(G,0)
    plt.show()