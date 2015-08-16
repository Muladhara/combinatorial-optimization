var x {j in 1..A}     >= 0 , <=1;
var y {j in 1..A+A+1}  >= 0 , <=1;

        maximize trasmissione: sum {j in 1..A}   c[j] * x[j];
          subject to knapsack: sum {j in 1..A}   w[j] * x[j] <= B;
 subject to tagli {i in 1..u}: sum {j in 1..A} V[i,j] * x[j] >= d[i];

# si possono aggiungere anche i vincoli (ridondanti) che impongono la presenza di almeno un arco incidente su ciascun nodo
# subject to nodi {i in 1..N}: sum {j in 1..A}M[i,j]* x[j]>=1; 

                      maximize Flusso: sum {i in 1..A+A+1}   wt[i]  *  y[i] ;
 subject to Incidenza {i in 1..N    }: sum {j in 1..A+A+1} Ms[i,j]  *  y[j] = domanda[i];
 subject to Capacita  {i in 1..A+A+1}:                        y[i] <= ct[i];
