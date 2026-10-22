#include "bf_decoding.h"
#include "gf2x_arith_mod_xPplusOne.h"
#include "bitslicing_helpers.h"

#include <bits/time.h>
#include <endian.h>
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/*
 * NOTA: il decoder è constant-time solo se compilato con -DCONSTANT_TIME=1.
 * Senza quella macro gf2x_mod_mul_monom usa word_level_shift_VT, che ha salti e
 * accessi in memoria che dipendono dalla quantità di rotazione (posizioni di H e
 * posizione flippata).
 */
#if !defined(CONSTANT_TIME)
#warning "bf_decoding.c compilato senza -DCONSTANT_TIME: la rotazione gf2x_mod_mul_monom NON è constant-time"
#endif

#define ROTBYTE(a)   ( (a << 8) | (a >> (DIGIT_SIZE_b - 8)) )
#define ROTUPC(a)   ( (a >> 8) | (a << (DIGIT_SIZE_b - 8)) )
#define ROUND_UP(amount, round_amt) ( ((amount+round_amt-1)/round_amt)*round_amt )

#define DELTA_LAYERS 3  // 3 bit per rappresentare [0..6]
#define N_REGS ((V + 7) / 8)
#define V_PADDED ((V + 7) & ~7)  // arrotonda al multiplo di 8 superiore

/* numero fisso di iterazioni: stesso limite di "iter < 1.5 * NUM_ERRORS_T" */
//#define BFMAX_ITERATIONS ((3 * NUM_ERRORS_T + 1) / 2)
#define BFMAX_ITERATIONS (NUM_ERRORS_T + 1)

/*#########################################################################################*/
/* UTILS FUNCTIONS                                                                         */
/*#########################################################################################*/

/* NON constant-time: scrive nella digit indicata da exponent. Il decoder ora usa
 * gf2x_toggle_coeff_CT; questa resta per eventuali usi con indici pubblici. */
static inline void gf2x_toggle_coeff(DIGIT poly[], const unsigned int exponent){
   /* Reverse the index, this because the polynomial is saved in big endian */
   int straightIdx = (NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b -1) - exponent;
   int digitIdx = straightIdx / DIGIT_SIZE_b;
   unsigned int inDigitIdx = straightIdx % DIGIT_SIZE_b;

   /* clear given coefficient */
   DIGIT mask = ( ((DIGIT) 1) << (DIGIT_SIZE_b-1-inDigitIdx));
   poly[digitIdx] = poly[digitIdx] ^ mask;
}

static inline void gf2x_copy(DIGIT dest[], const DIGIT in[]){
   int i = 0;

   for (; i <= NUM_DIGITS_GF2X_ELEMENT - 4; i += 4) {
       __m256i v = _mm256_load_si256((__m256i const*)(in + i));
       _mm256_store_si256((__m256i*)(dest + i), v);
   }

   for (; i < NUM_DIGITS_GF2X_ELEMENT; i++) {
       dest[i] = in[i];
   }
} // end gf2x_copy

static inline void gf2x_and(DIGIT Res[], const DIGIT A[], const DIGIT B[]){
   unsigned i;
   for (i = 0; i < NUM_DIGITS_GF2X_ELEMENT/4; i++) {
      __m256i a = _mm256_lddqu_si256((__m256i *)A + i);
      __m256i b = _mm256_lddqu_si256((__m256i *)B + i);
      _mm256_storeu_si256((__m256i *)Res + i, _mm256_and_si256(a, b));
   }
   for (i = i*4; i < NUM_DIGITS_GF2X_ELEMENT; i++) {
      Res[i] = A[i] & B[i];
   }
}

static inline void gf2x_xor(DIGIT Res[], const DIGIT A[], const DIGIT B[]){
    unsigned i;
    for (i = 0; i < NUM_DIGITS_GF2X_ELEMENT/4; i++) {
        __m256i a = _mm256_lddqu_si256((__m256i *)A + i);
        __m256i b = _mm256_lddqu_si256((__m256i *)B + i);
        _mm256_storeu_si256((__m256i *)Res + i, _mm256_xor_si256(a, b));
    }
    for (i = i*4; i < NUM_DIGITS_GF2X_ELEMENT; i++) {
        Res[i] = A[i] ^ B[i];
    }
}

/*#########################################################################################*/
/* CONSTANT-TIME HELPERS                                                                   */
/*#########################################################################################*/

/* 0xFF..FF se x != 0, 0 altrimenti. Il value barrier impedisce al compilatore di
 * trasformare le select successive in salti condizionati. */
static inline uint64_t ct_mask_nz64(uint64_t x){
   uint64_t m = (uint64_t)0 - ((x | ((uint64_t)0 - x)) >> 63);
   __asm__("" : "+r"(m));
   return m;
}

/* 0xFF..FF se a == b, 0 altrimenti */
static inline uint64_t ct_mask_eq64(uint64_t a, uint64_t b){
   return ~ct_mask_nz64(a ^ b);
}

/* a se m = 0xFF..FF, b se m = 0 */
static inline uint64_t ct_select64(uint64_t m, uint64_t a, uint64_t b){
   return (a & m) | (b & ~m);
}

/* 1 se x != 0, 0 altrimenti */
static inline uint32_t ct_is_nz64(uint64_t x){
   return (uint32_t)((x | ((uint64_t)0 - x)) >> 63);
}

/* numero di zeri finali di x (x != 0), senza tzcnt/bsf */
static inline uint32_t ct_ctz64(uint64_t x){
   uint64_t y = x & ((uint64_t)0 - x);          /* isola il bit a 1 più basso */
   uint32_t n = 0;
   n |= ct_is_nz64(y & 0xAAAAAAAAAAAAAAAAULL) << 0;
   n |= ct_is_nz64(y & 0xCCCCCCCCCCCCCCCCULL) << 1;
   n |= ct_is_nz64(y & 0xF0F0F0F0F0F0F0F0ULL) << 2;
   n |= ct_is_nz64(y & 0xFF00FF00FF00FF00ULL) << 3;
   n |= ct_is_nz64(y & 0xFFFF0000FFFF0000ULL) << 4;
   n |= ct_is_nz64(y & 0xFFFFFFFF00000000ULL) << 5;
   return n;
}

/*#########################################################################################*/
/* FUNCTIONS FOR SLICED COUNTER ARRAY STRUCTURE                                            */
/*#########################################################################################*/

/*
 * argmax constant-time sui contatori bitsliced.
 *
 * Restituisce block*P + j del contatore massimo; a parità sceglie l'indice globale
 * più basso (prima il blocco, poi l'esponente j), come gli argmax sui contatori uint8.
 *
 * Fase 1: dal piano più significativo al meno significativo, tiene solo i candidati
 *         che hanno quel bit a 1, se ne esiste almeno uno. Tutti i piani vengono
 *         sempre processati e la scelta è fatta con una maschera, non con un if.
 * Fase 2: per ogni blocco scorre tutte le digit e tiene l'ultima digit non nulla
 *         (layout big-endian: posizione fisica più alta = esponente più basso),
 *         poi il bit più basso al suo interno.
 */
static POSITION_T argmax_bitsliced_ct(const bs_operand_t bs[N0*NUM_SLICES_GF2X_ELEMENT]){

   __m256i candidate[N0*NUM_SLICES_GF2X_ELEMENT] __attribute__((aligned(32)));
   __m256i new_cand[N0*NUM_SLICES_GF2X_ELEMENT]  __attribute__((aligned(32)));

   /* candidati iniziali: tutti i bit validi. Esclusi i bit di slack in testa alla
    * digit 0 e le digit di padding oltre NUM_DIGITS_GF2X_ELEMENT (gli indici delle
    * digit sono pubblici) */
   for(int b = 0; b < N0; b++){
      DIGIT *c = (DIGIT *)&candidate[b*NUM_SLICES_GF2X_ELEMENT];
      for(int d = 0; d < NUM_SLICES_GF2X_ELEMENT*4; d++)
         c[d] = (d < NUM_DIGITS_GF2X_ELEMENT) ? ~(DIGIT)0 : (DIGIT)0;
      c[0] = SLACK_CLEAR_MASK;
   }

   /* FASE 1 */
   for(int i = BITSLICED_OPERAND_WIDTH-1; i >= 0; i--){
      __m256i any = _mm256_setzero_si256();
      for(int z = 0; z < N0*NUM_SLICES_GF2X_ELEMENT; z++){
         new_cand[z] = _mm256_and_si256(candidate[z], bs[z].slice[i]);
         any = _mm256_or_si256(any, new_cand[z]);
      }
      /* keep = 1..1 se almeno un candidato ha il bit i a 1 */
      uint64_t keep = ct_mask_nz64((uint64_t)(1 - _mm256_testz_si256(any, any)));
      __m256i vkeep = _mm256_set1_epi64x((long long)keep);
      for(int z = 0; z < N0*NUM_SLICES_GF2X_ELEMENT; z++)
         candidate[z] = _mm256_or_si256(_mm256_and_si256(new_cand[z], vkeep),
                                        _mm256_andnot_si256(vkeep, candidate[z]));
   }

   /* FASE 2: i blocchi sono scorsi dall'ultimo al primo, così vince il blocco più basso */
   uint64_t res_block = 0, res_j = 0;
   for(int b = N0-1; b >= 0; b--){
      const DIGIT *c = (const DIGIT *)&candidate[b*NUM_SLICES_GF2X_ELEMENT];
      uint64_t best_d = 0, best_w = 0, has = 0;
      for(uint64_t d = 0; d < NUM_DIGITS_GF2X_ELEMENT; d++){
         uint64_t nz = ct_mask_nz64(c[d]);
         best_d = ct_select64(nz, d, best_d);
         best_w = ct_select64(nz, c[d], best_w);
         has |= nz;
      }
      /* posizione fisica dal MSB del polinomio: digit*64 + (63 - indice del bit più basso);
       * se il blocco non ha candidati best_w = 0: "| (~has & 1)" evita ctz(0), il valore
       * viene comunque scartato dalla select */
      uint64_t phys = best_d*DIGIT_SIZE_b + (DIGIT_SIZE_b-1 - ct_ctz64(best_w | (~has & 1)));
      uint64_t j    = (uint64_t)NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b - 1 - phys;
      res_block = ct_select64(has, (uint64_t)b, res_block);
      res_j     = ct_select64(has, j, res_j);
   }
   return (POSITION_T)(res_block*P + res_j);
}

void lift_mul_dense_to_sparse_CT(bs_operand_t bs_res[], const DIGIT dense[], const POSITION_T sparse[], unsigned int nPos){
   SLICE_TYPE tmp[NUM_SLICES_GF2X_ELEMENT];
   /* gf2x_mod_mul_monom scrive solo NUM_DIGITS_GF2X_ELEMENT digit: senza questo memset
    * la parola di padding resta garbage dello stack, viene sommata V volte e produce
    * contatori uguali a V proprio nelle posizioni di padding */
   memset(tmp, 0, sizeof(tmp));
   for(unsigned int i =0; i< nPos; i++) {
#if (defined HIGH_PERFORMANCE_X86_64)
      gf2x_mod_mul_monom((DIGIT *)tmp,sparse[i],dense);
#else
      gf2x_mod_mul_monom(tmp,sparse[i],dense);
#endif
      for(int j = 0 ; j < NUM_SLICES_GF2X_ELEMENT; j++) {
         bs_res[j] = bitslice_inc(bs_res[j], tmp[j]);
      }
   }
}

/*#########################################################################################*/
/* CONSTANT-TIME UPDATES                                                                   */
/*#########################################################################################*/

/*
 * out[block] ^= x^exponent se enable = 0xFF..FF, nessun effetto se enable = 0.
 * Tocca tutte le N0*NUM_DIGITS_GF2X_ELEMENT digit e cambia solo quella giusta: l'indirizzo
 * scritto non dipende da block né da exponent. Accessi scalari: out può essere disallineato
 * (niederreiter_decrypt passa &decoded_error_vector[1]).
 */
static inline void gf2x_toggle_coeff_CT(DIGIT out[], uint64_t block, uint64_t exponent, uint64_t enable){
   uint64_t straightIdx = (uint64_t)NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b - 1 - exponent;
   uint64_t target      = block*NUM_DIGITS_GF2X_ELEMENT + straightIdx / DIGIT_SIZE_b;
   DIGIT bit            = ((DIGIT)1) << (DIGIT_SIZE_b - 1 - straightIdx % DIGIT_SIZE_b);
   for(uint64_t d = 0; d < (uint64_t)N0*NUM_DIGITS_GF2X_ELEMENT; d++)
      out[d] ^= bit & ct_mask_eq64(d, target) & enable;
}

/*
 * syndrome ^= x^exponent * HTr[block] se enable = 0xFF..FF.
 * HTr[block] viene selezionato con una maschera (si leggono tutti i blocchi), poi si fa
 * una sola rotazione constant-time.
 */
static inline void syndrome_update_CT(DIGIT syndrome[], const DIGIT HTr[N0][NUM_DIGITS_GF2X_ELEMENT],
                                      uint64_t block, uint64_t exponent, uint64_t enable){
   DIGIT htr_sel[NUM_DIGITS_GF2X_ELEMENT];
   DIGIT update[NUM_DIGITS_GF2X_ELEMENT];
   for(int d = 0; d < NUM_DIGITS_GF2X_ELEMENT; d++){
      DIGIT acc = 0;
      for(int b = 0; b < N0; b++) acc |= HTr[b][d] & ct_mask_eq64(block, (uint64_t)b);
      htr_sel[d] = acc;
   }
   gf2x_mod_mul_monom(update, (POSITION_T)exponent, htr_sel);
   for(int d = 0; d < NUM_DIGITS_GF2X_ELEMENT; d++)
      syndrome[d] ^= update[d] & enable;
}

/*#########################################################################################*/
/* DECODER                                                                                 */
/*#########################################################################################*/
int bf_decoding_CT(DIGIT out[], const POSITION_T HtrPosOnes[N0][V], const POSITION_T HPosOnes[N0][V], DIGIT privateSyndrome[]){

    /* Densify HTr (versione CT: nessuna scrittura a indirizzi che dipendono da H) */
    DIGIT HTr[N0][NUM_DIGITS_GF2X_ELEMENT] = {{0}};
    for(int i=0; i<N0; i++) {
        gf2x_mod_densify_CT(HTr[i],HtrPosOnes[i],V);
    }

    /* numero fisso di iterazioni; dopo la convergenza flip e update sono disattivati
     * dalla maschera active invece che da un'uscita anticipata dal ciclo */
    for(int iter = 0; iter < BFMAX_ITERATIONS; iter++){

        bs_operand_t bs[N0*NUM_SLICES_GF2X_ELEMENT];
        memset(bs, 0, sizeof(bs));

        for (int i = 0; i < N0; i++) {
            lift_mul_dense_to_sparse_CT(
                bs+(i*NUM_SLICES_GF2X_ELEMENT),
                privateSyndrome,
                HPosOnes[i],
                V
            );
        }

        POSITION_T flip = argmax_bitsliced_ct(bs);
        uint64_t block  = flip / P;   /* P costante: il compilatore usa una moltiplicazione */
        uint64_t x      = flip % P;

        /* active = 0xFF..FF finché la sindrome non è nulla */
        DIGIT synd_or = 0;
        for(int d = 0; d < NUM_DIGITS_GF2X_ELEMENT; d++) synd_or |= privateSyndrome[d];
        uint64_t active = ct_mask_nz64(synd_or);

        //gf2x_toggle_coeff_CT(out, block, x, active);
        
        gf2x_toggle_coeff(out + block * NUM_DIGITS_GF2X_ELEMENT, x);
        syndrome_update_CT(privateSyndrome, (const DIGIT (*)[NUM_DIGITS_GF2X_ELEMENT])HTr, block, x, active);
    }

    /* successo = sindrome nulla. Il controllo precedente
     *    while (check < NUM_DIGITS && privateSyndrome[check++] == 0);
     * restituiva successo anche quando era non nulla solo l'ultima digit */
    return population_count(privateSyndrome) == 0;
}