/*
 * t-test fixed-vs-random sui singoli componenti del decoder BF-max.
 *
 * Il file include direttamente bf_decoding.c per poter chiamare le funzioni static.
 * Il percorso si passa con -DBF_SOURCE='"percorso/bf_decoding.c"'.
 * Nel target CMake bf_decoding.c NON va compilato anche come sorgente separato.
 *
 * Uso: taskset -c 1 ./LEDAcrypt_KEM-CPA_SL_1_N0_2_ttest_components [fattore misure]
 *
 * Per ogni componente la chiave è fissa e le classi sono:
 *   fixed:  sempre lo stesso input segreto (stessa sindrome / stessa posizione)
 *   random: input segreto diverso a ogni misura
 *
 * Controprove (variabile d'ambiente MODE):
 *   MODE=ff  fixed vs fixed (due input fissi diversi)
 *   MODE=rr  random vs random (stessa distribuzione: un |t| alto qui è un artefatto
 *            dell'ambiente di misura, non un leakage)
 */
#ifndef BF_SOURCE
#error "definire BF_SOURCE con il percorso di bf_decoding.c"
#endif
#include BF_SOURCE

#include <math.h>
#include <stdlib.h>
#include <x86intrin.h>
#include "api.h"
#include "niederreiter.h"
#include "rng.h"

#define ND NUM_DIGITS_GF2X_ELEMENT
#define NSL (N0 * NUM_SLICES_GF2X_ELEMENT)

typedef struct { double n, mean, m2; } welford_t;
static void welford_push(welford_t *w, double x) {
   w->n += 1.0; double d = x - w->mean; w->mean += d / w->n; w->m2 += d * (x - w->mean);
}
static double welch_t(const welford_t *a, const welford_t *b) {
   double va = a->m2 / (a->n - 1.0), vb = b->m2 / (b->n - 1.0);
   return (a->mean - b->mean) / sqrt(va / a->n + vb / b->n);
}
static inline uint64_t cycles_start(void) { unsigned x; _mm_lfence(); uint64_t t = __rdtscp(&x); _mm_lfence(); return t; }
static inline uint64_t cycles_stop(void)  { unsigned x; uint64_t t = __rdtscp(&x); _mm_lfence(); return t; }
static int cmp_u64(const void *x, const void *y) {
   uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y; return (a > b) - (a < b);
}

/* risultato: max |t| tra i tagli 100/99/95/90 percentile */
static double analyse(const char *name, const uint64_t *t, const int *cls, long n) {
   static const double crop[4] = { 1.00, 0.99, 0.95, 0.90 };
   uint64_t *s = malloc(n * sizeof(uint64_t));
   memcpy(s, t, n * sizeof(uint64_t));
   qsort(s, n, sizeof(uint64_t), cmp_u64);
   double max_t = 0, mf = 0, mr = 0;
   for (int c = 0; c < 4; c++) {
      uint64_t cut = (crop[c] >= 1.0) ? UINT64_MAX : s[(long)(crop[c] * (n - 1))];
      welford_t w[2] = {{0}};
      for (long i = 0; i < n; i++) if (t[i] <= cut) welford_push(&w[cls[i]], (double)t[i]);
      double tv = fabs(welch_t(&w[0], &w[1]));
      if (tv > max_t) { max_t = tv; mf = w[0].mean; mr = w[1].mean; }
   }
   printf("  %-26s %8ld misure  fixed %10.1f  random %10.1f cicli  max|t| = %7.2f  %s\n",
          name, n, mf, mr, max_t, max_t > 4.5 ? "LEAKAGE" : "ok");
   free(s);
   return max_t;
}

int main(int argc, char *argv[]) {
   double scale = (argc > 1) ? atof(argv[1]) : 1.0;
   initialize_pseudo_random_generator_seed(0, NULL);

   unsigned char *pk = calloc(1, CRYPTO_PUBLICKEYBYTES), *sk = calloc(1, CRYPTO_SECRETKEYBYTES);
   unsigned char *ss = calloc(1, CRYPTO_BYTES), *ct = calloc(1, CRYPTO_CIPHERTEXTBYTES);
   crypto_kem_keypair(pk, sk);
   const privateKeyNiederreiter_t *key = (const privateKeyNiederreiter_t *) sk;

   DIGIT HTr[N0][ND] = {{0}};
   for (int i = 0; i < N0; i++) gf2x_mod_densify_CT(HTr[i], key->HtrPosOnes[i], V);

   /* Sorgenti degli input (lette solo durante la preparazione):
    *   classe fixed:  una sindrome e i suoi contatori
    *   classe random: POOL sindromi diverse e i loro contatori
    * Prima di ogni blocco di misure, l'input di ciascuna misura viene COPIATO in uno
    * slot di lavoro condiviso: il buffer usato non dipende dalla classe, quindi pagine,
    * insiemi di cache e allineamento sono gli stessi per le due classi (come in dudect).
    * Con la variabile d'ambiente MODE=ff o MODE=rr si ottengono le controprove
    * fixed-vs-fixed e random-vs-random. */
   enum { POOL = 32, BATCH = 32 };
   const char *mode = getenv("MODE") ? getenv("MODE") : "fr";
   static DIGIT src_synd[2][POOL][ND];
   static bs_operand_t src_bs[2][POOL][NSL];
   int src_n[2];
   for (int c = 0; c < 2; c++) {
      src_n[c] = (mode[c] == 'r') ? POOL : 1;
      for (int p = 0; p < src_n[c]; p++) {
         crypto_kem_enc(ct, ss, pk);
         gf2x_mod_mul_dense_to_sparse(src_synd[c][p], (const DIGIT *) ct, key->HtrPosOnes[N0-1], V);
      }
   }
   memset(src_bs, 0, sizeof src_bs);
   for (int c = 0; c < 2; c++)
      for (int p = 0; p < src_n[c]; p++)
         for (int i = 0; i < N0; i++)
            lift_mul_dense_to_sparse_CT(src_bs[c][p] + i * NUM_SLICES_GF2X_ELEMENT, src_synd[c][p], key->HPosOnes[i], V);

   static DIGIT slot_synd[BATCH][ND];
   static bs_operand_t slot_bs[BATCH][NSL];
   int b_cls[BATCH];
   uint64_t b_blk[BATCH], b_x[BATCH];

   long n_small = (long)(200000 * scale), n_lift = (long)(20000 * scale);
   long nmax = n_small > n_lift ? n_small : n_lift;
   uint64_t *t = calloc(nmax, sizeof(uint64_t));
   int *cls = calloc(nmax, sizeof(int));
   volatile uint64_t sink = 0;
   double worst = 0, r;

   printf("LEDA KEM-CPA cat %d, N0 = %d: t-test sui componenti (MODE = %s)\n", CATEGORY, N0, mode);

   /* classe e input di ogni slot; con copy_bs/copy_synd copia il contenuto nello slot */
#define PREPARE_BATCH(copy_bs, copy_synd)                                         \
   for (int k = 0; k < BATCH; k++) {                                              \
      int c = rand() & 1, p = rand() % src_n[c];                                  \
      b_cls[k] = c;                                                               \
      b_blk[k] = (mode[c] == 'r') ? (uint64_t)(rand() % N0) : 0;                  \
      b_x[k]   = (mode[c] == 'r') ? (uint64_t)(rand() % P)  : 0;                  \
      if (copy_bs)   memcpy(slot_bs[k], src_bs[c][p], sizeof slot_bs[k]);         \
      if (copy_synd) memcpy(slot_synd[k], src_synd[c][p], sizeof slot_synd[k]);   \
   }

   /* 1. argmax sui contatori */
   for (long i = 0; i < n_small; ) {
      PREPARE_BATCH(1, 0);
      for (int k = 0; k < BATCH && i < n_small; k++, i++) {
         uint64_t t0 = cycles_start();
         sink += argmax_bitsliced_ct(slot_bs[k]);
         t[i] = cycles_stop() - t0; cls[i] = b_cls[k];
      }
   }
   r = analyse("argmax_bitsliced_ct", t, cls, n_small); if (r > worst) worst = r;

   /* 2. aggiornamento dell'errore: fixed = (blocco 0, x 0), random = posizione casuale */
   static unsigned char ebuf[1 + N0 * ND * DIGIT_SIZE_B];
   for (long i = 0; i < n_small; ) {
      PREPARE_BATCH(0, 0);
      for (int k = 0; k < BATCH && i < n_small; k++, i++) {
         uint64_t blk = b_blk[k], x = b_x[k];
         uint64_t t0 = cycles_start();
         gf2x_toggle_coeff_CT((DIGIT *) &ebuf[1], blk, x, ~(uint64_t)0);
         t[i] = cycles_stop() - t0; cls[i] = b_cls[k];
      }
   }
   r = analyse("gf2x_toggle_coeff_CT", t, cls, n_small); if (r > worst) worst = r;

   /* 3. aggiornamento della sindrome */
   for (long i = 0; i < n_small; ) {
      PREPARE_BATCH(0, 1);
      for (int k = 0; k < BATCH && i < n_small; k++, i++) {
         uint64_t blk = b_blk[k], x = b_x[k];
         uint64_t t0 = cycles_start();
         syndrome_update_CT(slot_synd[k], (const DIGIT (*)[ND]) HTr, blk, x, ~(uint64_t)0);
         t[i] = cycles_stop() - t0; cls[i] = b_cls[k];
      }
   }
   r = analyse("syndrome_update_CT", t, cls, n_small); if (r > worst) worst = r;

   /* 4. lift (ricalcolo dei contatori) */
   static bs_operand_t bs_work[NSL];
   for (long i = 0; i < n_lift; ) {
      PREPARE_BATCH(0, 1);
      for (int k = 0; k < BATCH && i < n_lift; k++, i++) {
         memset(bs_work, 0, sizeof bs_work);
         uint64_t t0 = cycles_start();
         for (int b = 0; b < N0; b++)
            lift_mul_dense_to_sparse_CT(bs_work + b * NUM_SLICES_GF2X_ELEMENT, slot_synd[k], key->HPosOnes[b], V);
         t[i] = cycles_stop() - t0; cls[i] = b_cls[k];
         sink += (uint64_t)_mm256_extract_epi64(bs_work[0].slice[0], 0);
      }
   }
   r = analyse("lift (contatori)", t, cls, n_lift); if (r > worst) worst = r;

   printf("max |t| complessivo = %.2f -> %s\n", worst,
          worst > 4.5 ? "LEAKAGE in almeno un componente" : "nessuna differenza rilevata");
   (void) sink;
   free(pk); free(sk); free(ss); free(ct); free(t); free(cls);
   return 0;
}
