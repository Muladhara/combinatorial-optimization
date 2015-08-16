La consegna si compone di tre directory:

In fl09  viene implementato il Simplesso Dinamico per la soluzione del problema rilassato. Viene utilizzato il solutore cplex.
Il programma scrive in trace.txt la  traccia d'esecuzione, in trace_obj.txt il  debug oggetti, in trace_vincoli.txt il  debug vincoli e in fl10_data.dat la soluzione (da passare all'euristica). Usa i sottoprogrammi: fl09_dich.run, fl09_data.dat, fl09_modelli.mod fl09_verif.run, fl09_tagli.run, fl09_salva_vincolo.run dot_sol.run 

In fl10  si trova l'implementazione del Reverse Greedy con scambi alla fine che usa i sottoprogrammi: fl09_data.dat, fl10_dich.run, mk_structs.run, calcola_totali.run, dijkstra.run, elimina_arco.run, tieni_arco.run, rientro_nel_budget.run, sostituisci_arco.run, add_last.run, view_sol.run, load_distance_matrix.run, sostituisci_arco_x.run, dot_grafo.run
La soluzione viene scritta in fl10_sol.dat


In fl12  viene implementato il Reverse Greedy modificato con scambi.
Utilizza i sottoprogrammi: fl09_data.dat, fl10_dich.run,  mk_structs.run, calcola_totali.run, dijkstra.run, elimina_arco.run,ottieni_arco.run, rientro_nel_budget.run, sostituisci_arco.run, add_last.run, view_sol.run, load_distance_matrix.run,sostituisci_arco_x.run, dot_grafo.run 
La soluzione viene scritta in fl12_sol.dat


I programmi principali da lanciare sono gli flXX_main.run
