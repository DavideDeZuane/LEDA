/*
 * t-test di Welch fixed-vs-random (metodologia dudect) sul KEM-CPA di LEDA.
 *
 * Uso:
 *   taskset -c 1 ./LEDAcrypt_KEM-CPA_SL_1_N0_2_ttest decoder [misure]   tempo di bf_decoding_CT
 *   taskset -c 1 ./LEDAcrypt_KEM-CPA_SL_1_N0_2_ttest decaps  [misure]   tempo di crypto_kem_dec
 *
 * La chiave è fissa per tutto il test. Le due classi differiscono solo nel ciphertext:
 *   classe 0 (fixed):  sempre lo stesso ciphertext (stesso errore e)
 *   classe 1 (random): un ciphertext nuovo a ogni misura
 * L'ordine delle classi è casuale. Gli input (ciphertext e sindrome privata) vengono
 * preparati a blocchi prima delle misure, in buffer distinti anche per la classe fixed,
 * così le due classi eseguono lo stesso codice e usano la stessa quantità di memoria.
 *
 * Attenzione: sul decoder intero (decine di milioni di cicli) il rumore di misura è
 * grande e il test è poco sensibile. Per la verifica vera usare ttest_components.c,
 * che misura i singoli componenti.
 *
 * Interpretazione: |t| > 4.5 indica che il tempo dipende dall'input (leakage).
 * |t| < 4.5 significa solo che con quel numero di misure non si vede differenza:
 * più misure rendono il test più sensibile.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#include "api.h"
#include "niederreiter.h"
#include "bf_decoding.h"
#include "gf2x_arith_mod_xPplusOne.h"
#include "rng.h"

/* ---------------- statistiche online (Welford) ---------------- */
typedef struct { double n, mean, m2; } welford_t;
static void welford_push(welford_t *w, double x) {
   w->n += 1.0;
   double d = x - w->mean;
   w->mean += d / w->n;
   w->m2 += d * (x - w->mean);
}
static double welch_t(const welford_t *a, const welford_t *b) {
   double va = a->m2 / (a->n - 1.0), vb = b->m2 / (b->n - 1.0);
   return (a->mean - b->mean) / sqrt(va / a->n + vb / b->n);
}

/* ---------------- misura dei cicli ---------------- */
static inline uint64_t cycles_start(void) {
   unsigned aux; _mm_lfence(); uint64_t t = __rdtscp(&aux); _mm_lfence(); return t;
}
static inline uint64_t cycles_stop(void) {
   unsigned aux; uint64_t t = __rdtscp(&aux); _mm_lfence(); return t;
}
static int cmp_u64(const void *x, const void *y) {
   uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y;
   return (a > b) - (a < b);
}

/* come in dudect: si calcola t anche scartando le misure sopra alcuni percentili,
 * perché interrupt e context switch producono code lunghe che nascondono le differenze */
#define N_CROPS 4
static const double crop_pct[N_CROPS] = { 1.00, 0.99, 0.95, 0.90 };

int main(int argc, char *argv[]) {
   const char *mode = (argc > 1) ? argv[1] : "decoder";
   long n_meas = (argc > 2) ? atol(argv[2]) : 2000;
   int decoder_mode = (strcmp(mode, "decoder") == 0);
   if (!decoder_mode && strcmp(mode, "decaps") != 0) {
      fprintf(stderr, "modalità sconosciuta: %s (decoder | decaps)\n", mode);
      return EXIT_FAILURE;
   }

   initialize_pseudo_random_generator_seed(0, NULL);
   unsigned char *pk = calloc(1, CRYPTO_PUBLICKEYBYTES), *sk = calloc(1, CRYPTO_SECRETKEYBYTES);
   unsigned char *ss = calloc(1, CRYPTO_BYTES), *ss_dec = calloc(1, CRYPTO_BYTES);
   unsigned char *ct_fixed = calloc(1, CRYPTO_CIPHERTEXTBYTES), *ct = calloc(1, CRYPTO_CIPHERTEXTBYTES);
   uint64_t *t_all = malloc(n_meas * sizeof(uint64_t));
   int *cls_all = malloc(n_meas * sizeof(int));

   crypto_kem_keypair(pk, sk);
   crypto_kem_enc(ct_fixed, ss, pk);
   const privateKeyNiederreiter_t *key = (const privateKeyNiederreiter_t *) sk;

   DIGIT synd_fixed[NUM_DIGITS_GF2X_ELEMENT], synd[NUM_DIGITS_GF2X_ELEMENT];
   gf2x_mod_mul_dense_to_sparse(synd_fixed, (const DIGIT *) ct_fixed, key->HtrPosOnes[N0-1], V);
   unsigned char e[1 + N0*NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_B];
   volatile int sink = 0;

   /* riscaldamento: cache, frequenza della CPU */
   for (int w = 0; w < 10; w++) {
      memcpy(synd, synd_fixed, sizeof synd);
      sink += decoder_mode ? bf_decoding_CT((DIGIT *) &e[1], (const POSITION_T (*)[V]) key->HtrPosOnes,
                                            (const POSITION_T (*)[V]) key->HPosOnes, synd)
                           : crypto_kem_dec(ss_dec, ct_fixed, sk);
   }

   /* input preparati a blocchi PRIMA delle misure, in buffer distinti per entrambe le
    * classi: durante la misura le due classi eseguono lo stesso codice e occupano la
    * stessa quantità di memoria (la classe fixed usa copie dello stesso ciphertext) */
   enum { BATCH = 32 };
   static unsigned char ct_b[BATCH][CRYPTO_CIPHERTEXTBYTES];
   static DIGIT synd_b[BATCH][NUM_DIGITS_GF2X_ELEMENT];
   int cls_b[BATCH];
   int fails = 0;
   for (long i = 0; i < n_meas; ) {
      for (int k = 0; k < BATCH; k++) {
         cls_b[k] = rand() & 1;
         if (cls_b[k] == 0) memcpy(ct_b[k], ct_fixed, CRYPTO_CIPHERTEXTBYTES);
         else               crypto_kem_enc(ct_b[k], ss, pk);
         gf2x_mod_mul_dense_to_sparse(synd_b[k], (const DIGIT *) ct_b[k], key->HtrPosOnes[N0-1], V);
      }
      for (int k = 0; k < BATCH && i < n_meas; k++, i++) {
         memset(e, 0, sizeof e);
         uint64_t t0 = cycles_start();
         int r = decoder_mode ? bf_decoding_CT((DIGIT *) &e[1], (const POSITION_T (*)[V]) key->HtrPosOnes,
                                               (const POSITION_T (*)[V]) key->HPosOnes, synd_b[k])
                              : crypto_kem_dec(ss_dec, ct_b[k], sk);
         uint64_t t1 = cycles_stop();
         fails += decoder_mode ? (r != 1) : (r != 0);
         t_all[i] = t1 - t0;
         cls_all[i] = cls_b[k];
         if ((i + 1) % 500 == 0) fprintf(stderr, "\r  %ld/%ld misure", i + 1, n_meas);
      }
   }
   fprintf(stderr, "\n");

   /* soglie di taglio calcolate su tutte le misure */
   uint64_t *sorted = malloc(n_meas * sizeof(uint64_t));
   memcpy(sorted, t_all, n_meas * sizeof(uint64_t));
   qsort(sorted, n_meas, sizeof(uint64_t), cmp_u64);

   printf("modalità %s: %ld misure, %d decodifiche fallite, mediana %.3f Mcicli\n",
          mode, n_meas, fails, sorted[n_meas / 2] / 1e6);
   double max_t = 0;
   for (int c = 0; c < N_CROPS; c++) {
      uint64_t cut = (crop_pct[c] >= 1.0) ? UINT64_MAX : sorted[(long)(crop_pct[c] * (n_meas - 1))];
      welford_t w[2] = {{0}};
      for (long i = 0; i < n_meas; i++)
         if (t_all[i] <= cut) welford_push(&w[cls_all[i]], (double) t_all[i]);
      double t = fabs(welch_t(&w[0], &w[1]));
      if (t > max_t) max_t = t;
      printf("  taglio %3.0f%%: n = %6.0f / %6.0f, media fixed %.4f, random %.4f Mcicli, |t| = %6.2f\n",
             crop_pct[c] * 100, w[0].n, w[1].n, w[0].mean / 1e6, w[1].mean / 1e6, t);
   }
   printf("max |t| = %.2f -> %s\n", max_t,
          max_t > 4.5 ? "LEAKAGE: il tempo dipende dall'input"
                      : "nessuna differenza rilevata con queste misure");

   (void) sink;
   free(pk); free(sk); free(ss); free(ss_dec); free(ct_fixed); free(ct);
   free(t_all); free(cls_all); free(sorted);
   return EXIT_SUCCESS;
}
