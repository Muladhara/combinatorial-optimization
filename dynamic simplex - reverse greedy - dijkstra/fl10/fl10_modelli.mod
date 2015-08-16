var y {j in 1..A+A+1} >= 0 , <=1;

maximize Flusso:       sum {i in 1..A+A+1} wt[i] * y[i];
 subject to Incidenza {i in 1..N    }:  sum {j in 1..A+A+1} Ms[i,j] * y[j] = domanda[i];
 subject to Capacita  {i in 1..A+A+1}: y[i] <= ct[i];
