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

#define ROTBYTE(a)   ( (a << 8) | (a >> (DIGIT_SIZE_b - 8)) )
#define ROTUPC(a)   ( (a >> 8) | (a << (DIGIT_SIZE_b - 8)) )
#define ROUND_UP(amount, round_amt) ( ((amount+round_amt-1)/round_amt)*round_amt )

#define DELTA_LAYERS 3  // 3 bit per rappresentare [0..6]
#define N_REGS ((V + 7) / 8)
#define V_PADDED ((V + 7) & ~7)  // arrotonda al multiplo di 8 superiore

/*#########################################################################################*/
/* UTILS FUNCTIONS                                                                         */
/*#########################################################################################*/

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
/* FUNCTIONS FOR SLICED COUNTER ARRAY STRUCTURE                                            */
/*#########################################################################################*/

static inline int argmax_bitsliced(const bs_operand_t* bs, int n_slices_total) {
    
   // compute mask ok all 1 to found the max
   __m256i candidate[N0*NUM_SLICES_GF2X_ELEMENT] __attribute__((aligned(32)));
   //__m256i *candidate = aligned_alloc(32, n_slices_total * sizeof(__m256i));
   // most efficent way to set all register to 1
   for(int z = 0; z < N0*NUM_SLICES_GF2X_ELEMENT; z++) candidate[z] = _mm256_cmpeq_epi32(candidate[z], candidate[z]);

   /* FASE 1: scan MSB→LSB to select bits  */
   for(int i = BITSLICED_OPERAND_WIDTH-1; i >= 0; i--){
       uint32_t zero_ctr = 0;
       uint32_t nonzero_blocks = 0;

       for(int z = 0; z < N0*NUM_SLICES_GF2X_ELEMENT; z++){
         int is_zero = _mm256_testz_si256(candidate[z], candidate[z]);
         // controllo quali bit di candidate hanno un elemento in comune con bitsliced 
         zero_ctr += _mm256_testz_si256(candidate[z], bs[z].slice[i]);
         nonzero_blocks += !is_zero;
       }

       if(zero_ctr != N0*NUM_SLICES_GF2X_ELEMENT){
           for(int z = 0; z < N0*NUM_SLICES_GF2X_ELEMENT; z++)
               candidate[z] = _mm256_and_si256(candidate[z], bs[z].slice[i]);
      }
      if(nonzero_blocks == 1) break;
   }

   /* FASE 2: trova posizione fisica nel candidate */
   int phys_pos = -1;
   for(int i = 0; i < N0*NUM_SLICES_GF2X_ELEMENT && phys_pos == -1; i++){
       if(_mm256_testz_si256(candidate[i], candidate[i])) continue;

       uint64_t lanes[4];
       memcpy(lanes, &candidate[i], sizeof(lanes));
       for(int l = 0; l < 4 && phys_pos == -1; l++){
           if(lanes[l] != 0){
               int bit  = __builtin_clzll(lanes[l]);  // big-endian → clzll
               phys_pos = i * 256 + l * 64 + bit;
           }
       }
   }

   if(phys_pos == -1) return -1;

   /* FASE 3: converti posizione fisica → indice polinomio */
   int circulant_block = phys_pos / (NUM_SLICES_GF2X_ELEMENT * 256);
   int local_phys      = phys_pos % (NUM_SLICES_GF2X_ELEMENT * 256);
   int poly_idx        = local_phys - SLACK_SIZE;
   int j               = (P - 1) - poly_idx;          // inverti big-endian

   return circulant_block * P + j;
}


void lift_mul_dense_to_sparse_CT(bs_operand_t bs_res[], const DIGIT dense[], const POSITION_T sparse[], unsigned int nPos){
   SLICE_TYPE tmp[NUM_SLICES_GF2X_ELEMENT];
   for(int i =0; i< nPos; i++) {
#if (defined HIGH_PERFORMANCE_X86_64)
      /* note : last words of tmp will be intentionally garbage, in case
       * NUM_DIGITS_GF2X_ELEMENT is not divisible by 4, for alignment reasons
       * Their content won't be used */
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
/* DECODER                                                                                 */
/*#########################################################################################*/
int bf_decoding_CT(DIGIT out[], const POSITION_T HtrPosOnes[N0][V], const POSITION_T HPosOnes[N0][V], DIGIT privateSyndrome[]){

    /* Densify HTr */
    DIGIT HTr[N0][NUM_DIGITS_GF2X_ELEMENT] = {{0}};
    for(int i=0; i<N0; i++) {
        gf2x_mod_densify_CT(HTr[i],HtrPosOnes[i],V);
    }
    
    int iter = 0;
    int hw = population_count(privateSyndrome);
    DIGIT update[NUM_DIGITS_GF2X_ELEMENT];

    while (iter < 1.5 * NUM_ERRORS_T && hw != 0){

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

        POSITION_T flip = argmax_bitsliced(bs_unsatParityChecks, N0 * NUM_SLICES_GF2X_ELEMENT);
        int block    = flip / P;  // quale blocco di HTr
        int x        = flip % P;  // di quanto ruotare dentro quel blocco

        gf2x_mod_mul_monom(update, x, HTr[block]);
        gf2x_xor(privateSyndrome, update, privateSyndrome);
        gf2x_toggle_coeff(out + block * NUM_DIGITS_GF2X_ELEMENT, x);
        hw = population_count(privateSyndrome);
        iter++;

    }


   /* Check the solution of the decoder */
   int check = 0;
   while (check < NUM_DIGITS_GF2X_ELEMENT && privateSyndrome[check++] == 0);
   return (check == NUM_DIGITS_GF2X_ELEMENT);
   //return 1;
      
}
