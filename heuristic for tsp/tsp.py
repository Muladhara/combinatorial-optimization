MAX = 50


def diff(S1,S2):
    """Verifica funzione obiettivo"""
    return True/False


def _permutation_plus(Sin,3):
    """Restituisce 3 archi (i,j) non-adiacenti (ed eventualmente con condizioni limitative)"""
    for ...
        yield triplet


def _remove(S,triplet):
    """Elimina gli archi indicati in triplet da S"""
    
    return S

def _readd(S,triplet):
    """Riaggiunge gli archi che insistono sui nodi indicati da triplet a S tranne proprio triplet"""
    
    yield S

def vicinato(S):
    # ...
    Sin = S
    while True:
        L = len(Sin)
        for triplet in _permutation_plus(Sin,3):
            Sp = _remove(Sin,triplet)   # Soluzione iniziale senza la terna degli archi levati
            for Splus in _readd(Sp,triplet):                                   
                yield Splus
        

def localsearch(iniSol):
    Sin = iniSol

    while True:
        Sols = []
        for S in vicinato(Sin):
            if diff(S,Sin)>0:
                Sols.append(S)
                continue
        if len(Sols)>0:
            Sin = maxsol(Sols)
            continue
    
    # in Sin c'e la soluzione
    return Sin


def main():

    R = None
    multiStartSolutionArray = calculateStarts()
    for S in multiStartSolutionArray:
        S1 = localsearch(S)
        if diff(S1,R)>0:
            R=S1
    print R
