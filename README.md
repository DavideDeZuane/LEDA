# LEDA w/ BFmax

Implementation with BFmax.

Old decoder at `Optimized_Implementation/Common/library/bf_decoding.c.old`  
New decoder at `Optimized_Implementation/Common/library/bf_decoding.c`  

## ToDo

- [ ] Replace computing counters with bitslicing 

## Setup 

The first thing to do is include `libkeccack` 

Creare script con 

- clone del submodule
- compilare la librarie 
```
make x86-64/libXKCP.a EXTRA_CFLAGS="-march=native -mtune=native"
```

- sposta quello compilato e gli header nelle posizioni giuste

```
cd bin
sudo cp -r libXKCP.a.headers /usr/local/include/libkeccak.a.headers
sudo cp libXKCP.a /usr/local/lib/libkeccak.a
```
