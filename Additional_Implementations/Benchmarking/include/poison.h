#pragma once
/*
 * poison.h — marcatura dei dati segreti per il controllo constant-time con Valgrind.
 *
 * poison(addr, len)   : i byte diventano "non inizializzati" per memcheck. Ogni salto
 *                       condizionato o indirizzo di memoria calcolato a partire da quei
 *                       byte (anche indirettamente) viene segnalato da valgrind.
 * unpoison(addr, len) : i byte tornano "definiti" (da usare sui risultati che è lecito
 *                       rendere pubblici, prima di confrontarli o stamparli).
 *
 * Le macro sono attive solo compilando con -DCT_CHECK; altrimenti non generano codice,
 * quindi il file può restare incluso anche nei build normali.
 */

#if defined(CT_CHECK)
#include <valgrind/memcheck.h>
#define poison(addr, len)   VALGRIND_MAKE_MEM_UNDEFINED((addr), (len))
#define unpoison(addr, len) VALGRIND_MAKE_MEM_DEFINED((addr), (len))
#else
#define poison(addr, len)   ((void)(addr), (void)(len))
#define unpoison(addr, len) ((void)(addr), (void)(len))
#endif
