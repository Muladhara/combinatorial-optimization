#! python 
#! -*- coding: utf-8 -*-


# - Leggi il file 
# - Suddividi i nodi
# - Per ogni insieme di nodi
#   - Crea una soluzione semplice
#   - Parti dalla soluzione semplice e migliorala

import sys
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pprint import pprint
import random
import itertools
from collections import deque
from heapq import *

# Numero di sezioni se n>50
NUMERO_SEZIONI= 5
LIMITE_NODI   = np.inf
ORE_LAVORATIVE= 8.0

Lp = 5

def create_graph(M,first=0):
    (n,m) = M.shape
    print "LEN=",n,m
    _g = nx.Graph()
    _l = first
    for i in range(n):        
        _g.add_node(_l)
        _l += 1
    for i in range(n):
        for j in range(m):
            if i!=j:
                # print "ADD EDGE",i+first,j+first,M[i,j]
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
        #print node,l,tot,q[node]
        if Q + q[node]>C: # Nuovo Tour
            Tour.append(0)
            #print N,old,M[0,old]
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
    #print Ts

def christofides(G):
    """Restituisce il circuito Hamiltoniano"""
    pass

def suddividi_nodi(n,C,T,H,M,q):
    """Restituisce sotto Matrici"""
    if n<LIMITE_NODI:
        return ( (n,C,T,H,M,q,range(n)), )
    nxs = n / NUMERO_SEZIONI  # Numero medio di nodi per sezione
    SEZIONI = []
    sez     = []
    Ns      = list(H)
    Nl      = len(Ns)
    DIST_NODI={}
    for i,n in enumerate(Ns):
        if not DIST_NODI.has_key(n):
            DIST_NODI[n]=[]
        DIST_NODI[n].append(i)
    #print DIST_NODI
    i = 0
    #print len(DIST_NODI)
    for dist,num in sorted(DIST_NODI.items()):
        #print dist,num
        sez.extend(num)
        i+=len(num)
        if i>nxs:
            SEZIONI.append(sez)
            sez = []
            i=0
    SEZIONI.append(sez)
    #return SEZIONI
    R = []
    for i,s in enumerate(SEZIONI):
        P = (len(s),C,T,H[:,s],M[s][:,s],q[s],s)
        R.append(P)
    return R

def mksymmat(N,MAX):
    M = np.reshape(np.array([ np.random.randint(1,MAX) for i in range(0,N*N)]),newshape=(N,N))
    M = np.tril(M)
    D = np.tri(N,N,-1)
    M = np.multiply(M,D)
    M = np.add(M,M.transpose())
    return M

def m2str(M):
    (n,m)=M.shape
    strx = ""
    for i in range(n):
        strx+=' '.join([str(x) for x in list(M[i])])+"\n"
    return strx

def mkloadfile(fname,n,C,T,M,q):

    F = """Numero clienti n
%d
Capacita mezzo C
%d
Tempo di recupero monete T
%d
Tempi tij
%squantità da ritirare
%s
""" % (n,C,T,m2str(M),' '.join([ str(x) for x in q]))
    open(fname,'w').write(F)

def avantindietro(G,n,C,T,H,M,q):
    """Restituisce tutti i tour dal magazzino al nodo e viceversa"""
    nTours = n
    Tours = []
    for i in range(n):
        tour = [ -1, i, -1 ]
        Tours.append(tour)
    return Tours

def Random_Insertion(G,n,C,T,H,M,q):
    def _random_insertion(distance_vector):        
        n = np.inf
        i = 0
        while n==np.inf:
            i = random.randrange(len(distance_vector))
            n = distance_vector[i]
#            print "Estratto Casualmente il nodo",i,"distanza",n
        return i
    return _insertion_heuristic(G,n,C,T,H,M,q,_random_insertion)

def Nearest_Neighbor(G,n,C,T,H,M,q):
    def _nearest_neighbor(distance_vector):
        return distance_vector.index(min(distance_vector))
    return _insertion_heuristic(G,n,C,T,H,M,q,_nearest_neighbor)

def _insertion_heuristic(G,n,C,T,H,M,q,func):
    # Trova il più vicino dal magazzino
    Tours = []
    tour = []
    visited = []
    print "---------- INSERTION HEUIRSTIC"
    lH = list(H)
    n0 = func(lH)
    tour.append(-1)
    tour.append(n0)
    visited.append(n0)
    Q = q[n0]
    t = H[n0]+T
    _j = 0
    print "Il nodo più vicino è ",n0
    _np=n0
    for i in range(n-1):
        lH = [ x if j not in visited else np.inf for j,x in enumerate(M[n0].getA1())]
        # print "Il vettore delle distanze è:", lH
        n0 = func(lH)
        # print "Prossimo nodo:", n0
        # Posso appendere questo nodo al tour o devo creare un nuovo tour
        if Q+q[n0]>C or t+T+lH[n0]+H[n0]>ORE_LAVORATIVE*60.0: # Superata la capacità o la giornata di lavoro. Nuovo Tour
            if Q+q[n0]>C: print _j,"TOUR DI CAPACITA",Q, "--",Q+q[n0],">",C
            if t+T+lH[n0]+H[n0]>ORE_LAVORATIVE*60.0: print _j,"TOUR DI TEMPO",t+H[_np]," - ",t+T+lH[n0]+H[n0],">",ORE_LAVORATIVE*60.0
            _j += 1
            tour.append(-1)
            Tours.append(tour)
            tour = [ -1 ]
            Q = q[n0]
            t = H[n0]+T
        else:
            Q += q[n0]
            t += lH[n0] + T
        tour.append(n0)
        visited.append(n0)
        # print "STATUS:",Q,t+H[n0]
        _np=n0
    tour.append(-1)
    Tours.append(tour)
    print "---------- INSERTION HEURISTIC"
    return Tours

def tour_info(Tours,G,n,C,T,H,M,q):
    """Restituisce le informazioni estese su un tour
    In ingresso prende una lista di liste di nodi del tour"""
    TINFO = []
    i_T = 0.0
    i_Q = 0.0
    i_Z = 0.0
    for nt,tour in enumerate(Tours):
        TI = [ [ i for i in tour ], ]
        i_q = 0.0
        links = []
        last = tour[0]
        for node in tour[1:]:
            links.append((last,node))
            last = node            
            i_q += q[node] if node>=0 else 0.0
        TI.append(links)
        TI.append(i_q)
        i_z = 0.0
        i_t = 0.0
        beg = 1.0
        for nl,link in enumerate(links):
            if min(link)==-1:
                i_t += H[max(link)] + T*beg
                beg = 0.0
            else:
                i_t += M[link[0],link[1]] + T
#            print "link",nl,"T=",i_T,i_t,i_T%480

        TI.append(i_t)
        i_Z += i_t
        if (i_T % 480) + i_t > 480:
            print "COMPLETO GIORNATA TOUR",nt,"con ",480 - ( i_T % 480),"minuti perche",(i_T % 480) , "+",i_t , ">", 480
            i_t += 480.0 - ( i_T % 480.0)
        i_T += i_t
        i_Q += i_q
        TINFO.append(TI)
    print "Tempo di completamento dei ",len(TINFO),"tour",i_T,"min =",(i_T/60.0),"H su un netto di ",i_Z,"min =",(i_Z/60),"H"
    return (i_T,i_Q,TINFO,i_Z)

def main():
    R = None
    multiStartSolutionArray = calculateStarts()
    for S in multiStartSolutionArray:
        S1 = localsearch(S)
        if diff(S1,R)>0:
            R=S1
    #print R

def load_file(F):
    L = [l.rstrip() for l in open(F)]
    n = int(L[1])
    C = int(L[3])
    T = int(L[5])
    ML = L[7:7+n+1]
    q = np.array([ int(x) for x in L[7+n+2].split()])
    MS = ';\n'.join(ML)
    M = np.matrix(MS)
    return  (n,C,T,M,q)

def main_prog():
    SottoMatrici = suddividi_nodi(n,C,T,M,q)
    for n,C,T,M,q in SottoMatrici:
        BaseSolutions = soluzioni_semplici(n,C,T,M,q)        
    # ( (n',C',T',M',q'), ... , (...) )
    G = create_graph(M)  # Grafo completo
    Mminus = M[1:,1:] # Solo i clienti
    #print Mminus
    H = create_graph(Mminus,1) # Grafo dei clienti
    Mid = M[0:,0]
    #print Mid
    K = create_graph(Mid) # Grafo dal magazzino a ogni cliente
    T=nx.minimum_spanning_tree(H)
    nx.draw(T,pos=nx.spring_layout(H))
    construct(G,0)
    plt.show()

def problem_info(n,C,T,H,M,q,nodes):
    print "----------------------------------------"
    print "        Numero di Nodi: ", n
    print "          Capacità Max: ", C
    print "     Tempo di recupero: ", T
    print "Distanze dal Magazzino: ", ' '.join([ str(int(x)) for x in H ])
    print " Disponibilità ai Nodi: ", ' '.join([ str(x) for x in q ])
    print "        Nodi Originali: ", ' '.join([ str(x) for x in nodes ])

def select_matrix(tour,H,M):
    """Restituisce la matrice di adiacenza dei nodi del tour"""
    only_nodes = tour[1:-1]    
    h = np.mat( H[only_nodes] )
    m = M[only_nodes][:,only_nodes]
    m1 = np.concatenate((h.T,m),1)
    h1 = [0,]
    h1.extend(list(H[only_nodes]))
    h1 = np.mat(h1)
    sm = np.concatenate((h1,m1))
    return sm

def plus1(tour):
    return [ x+1 for x in tour ]

def near(x,y):
    return x==y-1 or x==y+1

def near3(x):
    return near(x[0],x[1]) or near(x[0],x[2]) or near(x[1],x[2])

def near3edges(e):
    for p in e:
        n0=p[0]
        n1=p[1]
#        if n0==-1 or n1==-1:
#            return True
        for q in e:
            if q[0]==n0 or q[1]==n0 or q[0]==n1 or q[1]==n1:
                if q[0]==n0 and q[1]==n1: # è lo stesso arco
                    continue
                else:
                    return True
    return False

def good_permutations(edges):
    good = []
    for p in itertools.permutations(edges,3):
        if near3edges(p):
            continue   
        good.append(p)
    return good

def all_nodes(edges):
    nodes = []
    a = [ nodes.extend(x) for x in edges ]
    return list(nodes)

def all_nodes_nodup(edges):
    nodes = []
    a = [ nodes.extend(x) for x in edges ]
    m = []
    a = [ m.append(x) for x in nodes if x not in m ]
    m.append(-1)
    return m

def all_nodes_nodup1(edges):
    nodes = []
    a = [ nodes.extend(x) for x in edges ]
    m = []
    a = [ m.append(x) for x in nodes if x not in m ]
    return m

def move3opt(nodes,G,M):
    pass

def opt_moves(edges):
    A = edges[0][0]
    B = edges[0][1]
    C = edges[1][0]
    D = edges[1][1]
    E = edges[2][0]
    F = edges[2][1]

    moves = (
        ( (A,B), (C,D), (E,F) ),
        ( (A,D), (B,E), (C,F) ),
        ( (A,F), (B,C), (D,E) ),
        ( (A,C), (B,F), (D,E) ),
        ( (A,F), (B,D), (C,E) ),
        ( (A,E), (B,C), (D,F) ),
        ( (A,C), (B,E), (D,F) ),
        ( (A,E), (B,D), (C,F) ),
        ( (A,D), (B,F), (C,E) ) )

    for move in moves:
        yield move



def edge_cost(move,M):
    C0 = M[move[0][0],move[0][1]]
    C1 = M[move[1][0],move[1][1]]
    C2 = M[move[2][0],move[2][1]]
    return C0+C1+C2

def elimina_archi(etour,old):
    extour = [ x for x in etour if x not in old ]
    return extour
        
def find_node(tour,n0):
    #print tour
    for t in tour:
        print "CERCO",n0, "in",t
        if t[0]==n0: 
            print "CANCELLO",t,tour.index(t)
            del tour[tour.index(t)]
            return t
        elif t[1]==n0:
            print "CANCELLO",t,tour.index(t)
            del tour[tour.index(t)]
            return (t[1],t[0])

def aggiungi_archi(n,tour,move):
    print "AGGIUNGO ARCHI TOUR=",tour
    print "MOSSE=",move
    # nodes = all_nodes_nodup1(move)
    # print "NODES",nodes
    # nd = [ x for x in tour if x[0] in nodes and x[1] in nodes ]
    # newn = nd # [ x for x in tour if x not in nd ]

    ctour = tour
    ctour.extend(move)
    T = []
    n0 = -1
    next = None
    while ctour or next==-1:
        next = find_node(ctour,n0)
        if next:
            T.append(next)
            n0 = next[1]
        else:
            return None

    return T

def tour_has(tour,edges):
    for e in edges:
        if e not in tour[1]:
            return False
    return True
        

def ThreeOpt(tour,G,n,C,T,H,M,q):
    (sn,se,sq,st) = tour
    good_edges = good_permutations(se)
    MINMIN = np.inf
    MINmove = None
    MINold  = None
    MINMINold  = None
    MINqueue = []
    PERM_h = []
    IMPROVEMENT = []
    for e in good_edges:
        if tour_has(tour,e):            
            COSTout = edge_cost(e,M)
            heappush(PERM_h,(1.0/COSTout,e))

    while PERM_h:
        c,e = heappop(PERM_h)
        nodes = all_nodes(e)
        MIN = c
        MOVE = np.nan
        i = 0
        print "---------- ",e
        for move in opt_moves(e):
            if not tour_has(tour,move):
                COST = edge_cost(move,M)
                if COST<MIN:
                    MOVE = move
                    MIN = COST
                    MINold = e
                print "PROVA MOSSA",move,COST,MIN,MINMIN
                heappush(MINqueue,(MIN,(MOVE,e)))
        #print "L good_edges",len(good_edges),len(MINqueue)
        new_tour = None
        while MINqueue :
            (MIN,(MINmove,MINold)) = heappop(MINqueue)
            if np.isnan(MINmove):
                return
            print "TOUR",se
            print "Trying MOVE", MINmove
            print "on OLD", MINold
            new_tour = elimina_archi(se,MINold)
            print "TOUR SENZA OLD",new_tour        
            new_tour = aggiungi_archi(n,new_tour,MINmove)
            
            if new_tour:
                print "Vecchio Tour",se
                print "Nuovo Tour",new_tour,all_nodes_nodup(new_tour)
                return all_nodes_nodup(new_tour)

def Ricerca_Locale_Tours(Tours,G,n,C,T,H,M,q):
    x_Tours = []
    print Tours
    for tour in Tours:
        print tour
        tour_ottimizzato = ThreeOpt(tour,G,n,C,T,H,M,q)
        if tour_ottimizzato:
            print "INSERT TOUR OTTIMIZZATO",tour_ottimizzato
            x_Tours.append(tour_ottimizzato)
        else:
            print "INSERT TOUR BASE",tour
            x_Tours.append(tour[0])
    return x_Tours

def view_tour(i,tour):
    print "TOUR",i,tour,
    print "T",tour[0]," C:",tour[2]," T:",(tour[3]/60.0),"H=",tour[3],"min"

def view_tours(i,tours):
    for tour in tours:
        print "TOUR",i,tour,
        print "T",tour[0]," C:",tour[2]," T:",(tour[3]/60.0),"H=",tour[3],"min"
        i+=1
    return i

def view_tour_info(Tour_Info):
    print "Time: ",Tour_Info[0]/60.0,"/",Tour_Info[0]/480,"---",Tour_Info[3]/60.0, "/",Tour_Info[3]/480
    print "Capacità Totale",Tour_Info[1]
    for i,tour in enumerate(Tour_Info[2]):
        view_tour(i,tour)

def view_tour_info_day(Tour_Info):
    print "Time: ",Tour_Info[0]/60.0,"/",Tour_Info[0]/480, "---",Tour_Info[3]/60.0,"/",Tour_Info[3]/480
    print "Capacità Totale",Tour_Info[1]
    j=0
    for i,tour in enumerate(Tour_Info[2]):
        if tour[1]:
            print "RESIDUO DELLA GIORNATA",i,tour[0],
            #print tour[1][0]
            print tour[1][0]
            j=view_tours(j,tour[1])
        else:
            print "GIORNO VUOTO",i


def reorder_tours_in_day(Tours):
    H = []
    not_visited=[ x for x in Tours]
    for i,tour in enumerate(Tours[2]):
        (sn,se,sq,st) = tour
        heappush(H,(tour[3],(i,tour)))
    NeoTours= [ [480.0,[]] for x in range(len(Tours[2])+1) ]
    i = 0
    i_T = 0.0
    i_Q = 0.0
    i_Z = 0.0
    Q = 0,0
    while H:
        (t,(j,T)) = heappop(H)
        [O,N] = NeoTours[i]
        if O>T[3]:
            NeoTours[i][0] = O-T[3]
            NeoTours[i][1].append(T)
            i_Q += T[2]
        else:
            i_T += 480
            print "Aggiungo un giorno",i_T
            i_Q += T[2]
            i += 1
            [O,N] = NeoTours[i]
            NeoTours[i][0] = O-T[3]
            NeoTours[i][1].append(T)
        i_Z += T[3]
    i_T += (480-NeoTours[i][0])
    print "Aggiungo il residuo",(480-NeoTours[i][0]),"=",i_T,i_T/60
    sol = (i_T,i_Q,NeoTours,i_Z)
    return sol

        
        
        # TODO

if __name__=="__main__":
    ##### GENERAZIONE FILE DEI DATI
    filename = 'input.txt'
    genera = False   #True genera il file random e lo chiama input, False prende in input il file

    if genera:
        N     = 100
        MAX   = 100    # Massima distanza tra i nodi
        MAXC  = 100   # Massima quantità nel nodo
        C_MAX = 200
        T_MAX = 15
        M = mksymmat(N+1,MAX)
        q = [  np.random.randint(1,MAXC) for x in range(0,N)]
        mkloadfile(filename,N,C_MAX,T_MAX,M,q)
    #
    #############################################################

    (n,C,T,M,q) = load_file(filename) 

    # Vettore delle distanze dal magazzino
    H = M[0].getA1()[1:]

    # Matrice dei soli nodi
    M = M[1:,1:]

    print "Num Nodi:%d\nC,T=%d,%d\nlM=%s\nlq=%d\nlH=%d" % (n,C,T,M.shape,len(q),H.size)

    S = suddividi_nodi(n,C,T,H,M,q)

    for s in [S[0],]:  # Se tutti sostituire con S
        print s
        (n,C,T,H,M,q,nodes) = s
        problem_info(n,C,T,H,M,q,nodes) 
        G = create_graph(M)

        # Tours = avantindietro(G,n,C,T,H,M,q)
        Tours = Nearest_Neighbor(G,n,C,T,H,M,q)
        #Tours = Random_Insertion(G,n,C,T,H,M,q)
        # Tours = Christofides(G,n,C,T,H,M,q)
        TI = tour_info(Tours,G,n,C,T,H,M,q)        
        print "TOUR INIZIALI ------------------------------ ",TI[0],TI[1]
        view_tour_info(TI)
        # sys.exit()
        # print Tours

        Tours_Finali = Ricerca_Locale_Tours(TI[2],G,n,C,T,H,M,q)

        print "TOUR FINALI ------------------------------ "
        Tour_Finali_Nodi = [ x for x in Tours_Finali]
        print Tour_Finali_Nodi
        TF = tour_info(Tour_Finali_Nodi,G,n,C,T,H,M,q)
        view_tour_info(TF)

        TD = reorder_tours_in_day(TF)
        print TD
        view_tour_info_day(TD)
        
