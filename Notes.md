
## Sottrazione Condizionale



### Step 2 — la negazione unaria `-(…)`
```
-(0) = 0x0000000000000000   → nessuna correzione necessaria
-(1) = 0xFFFFFFFFFFFFFFFF   → tutti 1, correzione necessaria
```

Il complemento a due di `1` su 64 bit è esattamente `0xFFFF...FFFF`. Questa è la maschera.

### Step 3 — `CODE_REDUNDANCY & mask`
```
CODE_REDUNDANCY & 0xFFFF...FFFF = CODE_REDUNDANCY   (sottraiamo r)
CODE_REDUNDANCY & 0x0000...0000 = 0                 (sottraiamo 0)
```






Modifiche da fare con gcc version 15:

Il problema è una combinazione di due cose:
1. Arch usa GCC 15 che ha un linker più strict rispetto alle versioni precedenti. Con le librerie statiche .a, GCC 15 richiede che la libreria compaia dopo gli object file che la usano nella riga di linking:
# GCC vecchio (Debian) → funzionava in qualsiasi ordine
gcc main.o -lkeccak -lm

# GCC 15 (Arch) → la libreria statica DEVE stare dopo chi la usa
gcc main.o -lm /usr/local/lib/libkeccak.a  ← ordine obbligatorio


Quindi c'è da cambiare 

problema è che KECCAK_EXTERNAL_LIB contiene il path assoluto /usr/local/lib/libkeccak.a ma il linker con le librerie statiche su Arch richiede che venga dopo gli object file.
Sostituisci entrambe le righe target_link_libraries con questa forma che forza il path assoluto:
cmaketarget_link_libraries(${TARGET_BINARY_NAME} m ${SANITIZE} /usr/local/lib/libkeccak.a)


Hai ragione, allora la spiegazione è più semplice. Quando hai messo il path assoluto direttamente:
cmaketarget_link_libraries(${TARGET_BINARY_NAME} m ${SANITIZE} /usr/local/lib/libkeccak.a)
CMake con il path assoluto gestisce la libreria statica in modo diverso rispetto a quando passi una variabile — la tratta come un file da linkare esplicitamente e la posiziona correttamente nella riga del linker dopo gli object file automaticamente.
Invece con:
cmaketarget_link_libraries(${TARGET_BINARY_NAME} m ${SANITIZE} ${KECCAK_EXTERNAL_LIB})
Anche se KECCAK_EXTERNAL_LIB contiene lo stesso path, CMake su alcune versioni risolve la variabile in modo diverso e la libreria finisce in una posizione sbagliata nella riga del linker.
In pratica è un comportamento di CMake — path letterale e variabile contenente lo stesso path non vengono trattati identicamente dal generatore di Makefile. Su Debian con GCC più vecchio questo non causava problemi perché il linker era più permissivo, su Arch con GCC 15 il comportamento strict lo ha reso evidente.
