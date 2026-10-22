/*
 * Controllo constant-time del KEM-CPA con Valgrind (memcheck) e poison.h.
 *
 * Uso:
 *   valgrind ./LEDAcrypt_KEM-CPA_SL_1_N0_2_ctcheck decoder   solo bf_decoding_CT
 *   valgrind ./LEDAcrypt_KEM-CPA_SL_1_N0_2_ctcheck decaps    intera crypto_kem_dec
 *
 * Ogni messaggio di valgrind indica un punto in cui il codice dipende da un segreto:
 *   "Conditional jump or move depends on uninitialised value(s)"  -> salto sul segreto
 *   "Use of uninitialised value of size 8"                         -> indirizzo dal segreto
 *   "Use of uninitialised value of size 1"                         -> di solito uno shift
 *                                                                     con quantità segreta
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api.h"
#include "niederreiter.h"
#include "bf_decoding.h"
#include "gf2x_arith_mod_xPplusOne.h"
#include "rng.h"
#include "poison.h"

#if !defined(CT_CHECK)
#error "compilare con -DCT_CHECK, altrimenti poison/unpoison non hanno effetto"
#endif

int main(int argc, char *argv[])
{
   const char *mode = (argc > 1) ? argv[1] : "decoder";
   int runs = (argc > 2) ? atoi(argv[2]) : 1;

   initialize_pseudo_random_generator_seed(0, NULL);

   unsigned char *pk       = calloc(1, CRYPTO_PUBLICKEYBYTES),
                 *sk       = calloc(1, CRYPTO_SECRETKEYBYTES),
                 *ss       = calloc(1, CRYPTO_BYTES),
                 *decap_ss = calloc(1, CRYPTO_BYTES),
                 *ct       = calloc(1, CRYPTO_CIPHERTEXTBYTES);
   int failures = 0;

   for (int r = 0; r < runs; r++) {
      /* keygen ed encaps fuori dal controllo: qui interessa la decapsulazione */
      crypto_kem_keypair(pk, sk);
      crypto_kem_enc(ct, ss, pk);
      privateKeyNiederreiter_t *key = (privateKeyNiederreiter_t *) sk;

      if (strcmp(mode, "decoder") == 0) {
         /* sindrome privata calcolata come in decrypt_niederreiter_indcpa,
          * PRIMA di avvelenare: così si controlla solo il decoder */
         DIGIT privateSyndrome[NUM_DIGITS_GF2X_ELEMENT];
         gf2x_mod_mul_dense_to_sparse(privateSyndrome, (const DIGIT *) ct,
                                      key->HtrPosOnes[N0-1], V);
         unsigned char e[1 + N0*NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_B] = {0};

         /* segreti: chiave privata e sindrome (dipende dall'errore) */
         poison(key, sizeof(*key));
         poison(privateSyndrome, sizeof(privateSyndrome));

         int ok = bf_decoding_CT((DIGIT *) &e[1],
                                 (const POSITION_T (*)[V]) key->HtrPosOnes,
                                 (const POSITION_T (*)[V]) key->HPosOnes,
                                 privateSyndrome);

         /* risultati resi pubblici solo per stamparli */
         unpoison(&ok, sizeof(ok));
         unpoison(key, sizeof(*key));
         failures += (ok != 1);
      } else if (strcmp(mode, "decaps") == 0) {
         poison(sk, CRYPTO_SECRETKEYBYTES);

         int ret = crypto_kem_dec(decap_ss, ct, sk);

         unpoison(&ret, sizeof(ret));
         unpoison(decap_ss, CRYPTO_BYTES);
         unpoison(sk, CRYPTO_SECRETKEYBYTES);
         failures += (ret != 0) || memcmp(ss, decap_ss, CRYPTO_BYTES) != 0;
      } else {
         fprintf(stderr, "modalità sconosciuta: %s (decoder | decaps)\n", mode);
         return EXIT_FAILURE;
      }
   }
   printf("modalità %s: %d esecuzioni, %d fallimenti\n", mode, runs, failures);

   free(pk); free(sk); free(ss); free(decap_ss); free(ct);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
