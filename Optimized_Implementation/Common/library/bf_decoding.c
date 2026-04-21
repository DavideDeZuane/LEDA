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

#define ROTBYTE(a)   ( (a << 8) | (a >> (DIGIT_SIZE_b - 8)) )
#define ROTUPC(a)   ( (a >> 8) | (a << (DIGIT_SIZE_b - 8)) )
#define ROUND_UP(amount, round_amt) ( ((amount+round_amt-1)/round_amt)*round_amt )
#define SIGMA_LEN ((N0*P + 31) & ~31) 

#define DELTA_LAYERS 3  // 3 bit per rappresentare [0..6]


static inline
void gf2x_toggle_coeff(DIGIT poly[], const unsigned int exponent)
{
   /* Reverse the index, this because the polynomial is saved in big endian */
   int straightIdx = (NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b -1) - exponent;
   int digitIdx = straightIdx / DIGIT_SIZE_b;
   unsigned int inDigitIdx = straightIdx % DIGIT_SIZE_b;

   /* clear given coefficient */
   DIGIT mask = ( ((DIGIT) 1) << (DIGIT_SIZE_b-1-inDigitIdx));
   poly[digitIdx] = poly[digitIdx] ^ mask;
}

static inline void gf2x_copy(DIGIT dest[], const DIGIT in[])
{
   int i = 0;

   for (; i <= NUM_DIGITS_GF2X_ELEMENT - 4; i += 4) {
       __m256i v = _mm256_load_si256((__m256i const*)(in + i));
       _mm256_store_si256((__m256i*)(dest + i), v);
   }
   
   for (; i < NUM_DIGITS_GF2X_ELEMENT; i++) {
       dest[i] = in[i];
   }
} // end gf2x_copy

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

static inline void gf2x_xor(DIGIT Res[], const DIGIT A[], const DIGIT B[])
{
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

POSITION_T argmax_avx2(const uint8_t* arr, size_t len) {

    /* ------------------------------------------------- */
    /*  Find Max                                         */
    /* ------------------------------------------------- */
    POSITION_T i = 0;
    __m256i max_vec = _mm256_setzero_si256();
    for (; i <= len - 32; i += 32) {
        __m256i v = _mm256_loadu_si256((__m256i*)&arr[i]);
        max_vec = _mm256_max_epu8(max_vec, v);
    }
    
    __m256i v = max_vec;
    /* Horizontal Reduction */
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i m  = _mm_max_epu8(lo, hi);

    m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
    uint8_t max = (uint8_t)_mm_extract_epi8(m, 0);

    // remaining positions
    for (; i < len; i++) { if (arr[i] > max) max = arr[i]; }

    /* ------------------------------------------------- */
    /*  Find Argmax                                      */
    /* ------------------------------------------------- */
    __m256i vmax = _mm256_set1_epi8(max);

    // load 32 element from the array 
    // blocchi AVX2
    for (i = 0; i <= len - 32; i += 32) {
        __m256i v = _mm256_load_si256((__m256i*)(arr + i));
        // comparison between bytes of the two arrays, the result of the comparison is stored inside the mask v[j] == max_val → mask[j] = 0xFF, altrimenti 0x00. 
        // inside the mask register we have the index of the maximum 
        __m256i mask = _mm256_cmpeq_epi8(v, vmax);
        // starting from the mask create a bitmask, take the MSB of each byte and palce inside a 32 bit bits = b31b30...b1b0
        int bits = _mm256_movemask_epi8(mask);
        //__builtin_ctz(bits) → “count trailing zeros”, cioè quanti zeri ci sono prima del primo 1 nel numero binario. so adding 1 we find the maxium value global iondex
        // modificare per aggiungere clzll
        if (bits) return i + __builtin_ctz(bits);
    }

    // residual positions
    for (i = (len / 32) * 32; i < len; i++) { if (arr[i] == max) return i; }

    return -1;
}

/**
 * @brief Reference function for compute counters
 */
static inline void compute_counters_plain(uint8_t* sigma, DIGIT H[N0][NUM_DIGITS_GF2X_ELEMENT], DIGIT s[]){
   
      DIGIT h[NUM_DIGITS_GF2X_ELEMENT] = {0};
      DIGIT tmp[NUM_DIGITS_GF2X_ELEMENT] = {0};

      /* FULL SCAN OF THE MATRIX H, THE COLUMN hi IS COMPUTEND IN AN INCREMENTAL WAY */
      for(int i=0; i<N0; i++){
         POSITION_T offset = i * P;
         /* COPYING CIRCULANT OF THE BLOCK*/
         memcpy(h, H[i], DIGIT_SIZE_B * NUM_DIGITS_GF2X_ELEMENT);
         
         for(int j=0; j < P; j++){
            // in teoria la riscrivo tutta quindi non c'è bisogno 
            gf2x_and(tmp, h, s);
            sigma[j + offset] = population_count(tmp);
            /* */
            gf2x_mod_mul_monom(h,  1, h);
         }
      }
}

void compute_counters_sliced(const bs_operand_t* bs, uint8_t* ctrs, int total_elements, int bitsliced_width) 
{
   memset(ctrs, 0, total_elements * sizeof(uint8_t));

   for (int i = 0; i < N0; i++) {
      // global offset 
      const bs_operand_t* bs_block = bs + i * NUM_SLICES_GF2X_ELEMENT;
      
      for (int j = 0; j < P; j++) {

         // inversione dell'ordine dato che il polinomio è rappresentato in big endian mentre il counter array considera le posizioni in little endian
         int poly_idx    = (P - 1) - j;
         // tocca aggiungere il padding di 61 bit che si trova all'inizio
         int adjusted    = poly_idx + SLACK_SIZE;
         int block       = adjusted / 256;
         int lane        = (adjusted / 64) % 4;
         int bit_in_lane = 63 - (adjusted % 64);

         uint64_t lanes[4];
         uint8_t val = 0;
         for (int k = 0; k < bitsliced_width; k++) {
            memcpy(lanes, &bs_block[block].slice[k], sizeof(lanes));
            uint64_t bit = (lanes[lane] >> bit_in_lane) & 1ULL;
            val += (uint8_t)(bit << k);
         }
         ctrs[i * P + j] = val;
      }
  }
}

static inline void update_counters_bitsliced(
    bs_operand_t     bs_unsatParityChecks[N0*NUM_SLICES_GF2X_ELEMENT],
    const POSITION_T HtrPosOnes[N0][V],
    const DIGIT      H_dense[N0][NUM_DIGITS_GF2X_ELEMENT],
    const DIGIT      syndrome[],
    POSITION_T       pos_flip
) {
  
        #define N_REGS ((V + 7) / 8)
    
        int b = pos_flip >= P ? 1 : 0;
        POSITION_T local_pos = pos_flip - b * P;
    
        __m256i vp    = _mm256_set1_epi32((uint32_t)P);
        __m256i vpos  = _mm256_set1_epi32((uint32_t)local_pos);
    
        // pre-carica HtrPosOnes[b] nei registri
        __m256i htr_regs[N_REGS];
        for (int r = 0; r < N_REGS; r++) {
            uint32_t tmp[8] = {0};
            for (int i = 0; i < 8 && r*8+i < V; i++)
                tmp[i] = HtrPosOnes[b][r*8+i];
            htr_regs[r] = _mm256_loadu_si256((__m256i *)tmp);
        }
    
        // calcola tutti i row_index in parallelo
        POSITION_T row_indices[V];
        for (int r = 0; r < N_REGS; r++) {
            __m256i sum = _mm256_add_epi32(htr_regs[r], vpos);
            __m256i sub = _mm256_sub_epi32(sum, vp);
            __m256i msk = _mm256_cmpgt_epi32(vp, sum);
            __m256i res = _mm256_blendv_epi8(sub, sum, msk);
    
            uint32_t tmp[8];
            _mm256_storeu_si256((__m256i *)tmp, res);
            for (int i = 0; i < 8 && r*8+i < V; i++)
                row_indices[r*8+i] = tmp[i];
        }
    
        // leggi i segni dalla sindrome per tutti i row_index
        int ds[V];
        for (int i = 0; i < V; i++)
            ds[i] = gf2x_get_coeff(syndrome, row_indices[i]) ? 1 : -1;
    
        // aggiorna i counter: separa inc e dec per minimizzare le chiamate
        SLICE_TYPE tmp_slice[NUM_SLICES_GF2X_ELEMENT];
    
        for (int b2 = 0; b2 < N0; b2++) {
            bs_operand_t *bs_block = bs_unsatParityChecks + b2 * NUM_SLICES_GF2X_ELEMENT;
    
            for (int i = 0; i < V; i++) {
                gf2x_mod_mul_monom((DIGIT *)tmp_slice, row_indices[i], H_dense[b2]);
    
                if (ds[i] == 1) {
                    for (int j = 0; j < NUM_SLICES_GF2X_ELEMENT; j++)
                        bs_block[j] = bitslice_inc(bs_block[j], tmp_slice[j]);
                } else {
                    for (int j = 0; j < NUM_SLICES_GF2X_ELEMENT; j++)
                        bs_block[j] = bitslice_dec(bs_block[j], tmp_slice[j]);
                }
            }
        }
    
    
}

static inline void update_counters_after_flip(uint8_t *sigma, const POSITION_T H[N0][V], POSITION_T pos_flip, DIGIT* syndrome) 
{
    int b = pos_flip >= P ? 1 : 0;
    POSITION_T local_pos = pos_flip - b * P;

    POSITION_T row_indices[V];
    int ds[V];

    // calcolo row_indices e ds dalla sindrome già aggiornata
    for (int i = 0; i < V; i++) {
        POSITION_T row_index = H[b][i] + local_pos;
        if (row_index >= P) row_index -= P;
        row_indices[i] = row_index;
        ds[i] = gf2x_get_coeff(syndrome, row_index) ? 1 : -1;
    }

    // aggiorna i counter
    for (int i = 0; i < V; i++) {
        POSITION_T row_index = row_indices[i];
        int d = ds[i];
        for (int b2 = 0; b2 < 2; b2++) {
            POSITION_T offset = b2 * P;
            const POSITION_T *h2 = H[b2];
            POSITION_T ells[V];
            for (int j = 0; j < V; j++) {
                ells[j] = row_index - h2[j] + P;
                if (ells[j] >= P) ells[j] -= P;
            }
            for (int j = 0; j < V; j++)
                sigma[offset + ells[j]] += d;
        }
    }
}


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


static inline int argmax_bitsliced_impv(const bs_operand_t* bs, int n_slices_total) {

    //__m256i *candidate = aligned_alloc(32, N0*NUM_SLICES_GF2X_ELEMENT * sizeof(__m256i));
   __m256i candidate[N0*NUM_SLICES_GF2X_ELEMENT] __attribute__((aligned(32)));
   for(int z = 0; z < N0*NUM_SLICES_GF2X_ELEMENT; z++)
       candidate[z] = _mm256_cmpeq_epi32(candidate[z], candidate[z]);

   /* phase 1: scan MSB→LSB with early exit */
   int active_blocks = N0*NUM_SLICES_GF2X_ELEMENT;
   for(int i = BITSLICED_OPERAND_WIDTH-1; i >= 0; i--){
        __m256i new_cand[N0*NUM_SLICES_GF2X_ELEMENT] __attribute__((aligned(32)));     
       uint32_t any_set = 0;

       for(int z = 0; z < N0*NUM_SLICES_GF2X_ELEMENT; z++){
           new_cand[z] = _mm256_and_si256(candidate[z], bs[z].slice[i]);
           any_set    |= !_mm256_testz_si256(new_cand[z], new_cand[z]);
       }

       if(any_set){
           active_blocks = 0;
           for(int z = 0; z < N0*NUM_SLICES_GF2X_ELEMENT; z++){
               candidate[z] = new_cand[z];
               active_blocks += !_mm256_testz_si256(candidate[z], candidate[z]);
           }
           if(active_blocks == 1) break;  // early exit
       }
   }

   /* phase 2: find physical location of the argmax */
   int phys_pos = -1;
   for(int i = 0; i < N0*NUM_SLICES_GF2X_ELEMENT && phys_pos == -1; i++){
       if(_mm256_testz_si256(candidate[i], candidate[i])) continue;

       uint64_t lanes[4];
       memcpy(lanes, &candidate[i], sizeof(lanes));
       for(int l = 0; l < 4 && phys_pos == -1; l++){
           if(lanes[l] != 0){
               int bit  = __builtin_clzll(lanes[l]);   // big-endian → clzll this is needed because the counter array are stored from the MSB to the LSB
               phys_pos = i * 256 + l * 64 + bit;
           }
       }
   }


   if(phys_pos == -1) return -1;

   /* phase 3: conversion from physical position to polynomio index */
   int circulant_block = phys_pos / (NUM_SLICES_GF2X_ELEMENT * 256);
   int local_phys      = phys_pos % (NUM_SLICES_GF2X_ELEMENT * 256);
   int poly_idx        = local_phys - SLACK_SIZE;
   int j               = (P - 1) - poly_idx;

   return circulant_block * P + j;
}


/**
 * @brief BFmax decoder
 *    
 * @todo Capire quale parametro è quello giusto tra le rappresentazioni di H
 */
int bf_decoding_CT(DIGIT out[], const POSITION_T HtrPosOnes[N0][V], const POSITION_T HPosOnes[N0][V], DIGIT privateSyndrome[])
{

   /* Densify h_0, h_1, ..., h_n0-1 */
    DIGIT HTr[N0][NUM_DIGITS_GF2X_ELEMENT] = {{0}};
    for(int i=0; i<N0; i++) {
        gf2x_mod_densify_VT(HTr[i],HtrPosOnes[i],V);
    }

    DIGIT H_dense[N0][NUM_DIGITS_GF2X_ELEMENT] = {{0}};
    for(int i = 0; i < N0; i++)
        gf2x_mod_densify_VT(H_dense[i], HPosOnes[i], V);
    /* In this way we can update the counter as a xor between syndrome and h_i*/

    int iter = 0;
    int hw = population_count(privateSyndrome);
    DIGIT update[NUM_DIGITS_GF2X_ELEMENT] = {0};
    bs_operand_t bs_unsatParityChecks[N0*NUM_SLICES_GF2X_ELEMENT];
    memset(bs_unsatParityChecks, 0, sizeof(bs_unsatParityChecks));
    
    for (int i = 0; i < N0; i++) {
         lift_mul_dense_to_sparse_CT(bs_unsatParityChecks+(i*NUM_SLICES_GF2X_ELEMENT),
        privateSyndrome,
        HPosOnes[i],
        V);
      }

    uint8_t sigma[N0*P] __attribute__((aligned(32)));
    memset(sigma, 0, N0*P*sizeof(uint8_t));
    //compute_counters_sliced(bs_unsatParityChecks, sigma, N0*P, BITSLICED_OPERAND_WIDTH);
      
   do{
      //memset(sigma, 0, N0*P*sizeof(uint8_t));
      //memset(bs_unsatParityChecks, 0, sizeof(bs_unsatParityChecks));
      memset(update, 0, DIGIT_SIZE_B*NUM_DIGITS_GF2X_ELEMENT);
      
      /* COMPUTING COUNTERS BITSLICED */
      //for (int i = 0; i < N0; i++) {
       //  lift_mul_dense_to_sparse_CT(bs_unsatParityChecks+(i*NUM_SLICES_GF2X_ELEMENT),
         //   privateSyndrome,
           // HPosOnes[i],
            //V);
      //}
   
      //compute_counters_sliced(bs_unsatParityChecks, sigma, N0*P, BITSLICED_OPERAND_WIDTH);
      //compute_counters_be(bs_unsatParityChecks, sigma, N0 * P, BITSLICED_OPERAND_WIDTH);
      /* In gf2x_and possiamo evitare di considerare l'endianess e farlo diretto tanto usano la stessa rappresentazione */

      /* SCHOOLBOOK APPROACH FOR COMPUTING COUNTERS*/
      //compute_counters_sliced(bs_unsatParityChecks, sigma, N0*P, BITSLICED_OPERAND_WIDTH);
      //POSITION_T flip = argmax_avx2(sigma, N0*P);
      POSITION_T flip = argmax_bitsliced_impv(bs_unsatParityChecks, N0 * NUM_SLICES_GF2X_ELEMENT);
      /* ---------------------------------- */

      /* FIND POSITION TO FLIP */

      int block    = flip / P;  // quale blocco di HTr
      int x        = flip % P;  // di quanto ruotare dentro quel blocco
      // the position is in little endian so it's ok because the conversion is done by the function
      gf2x_toggle_coeff(out + block * NUM_DIGITS_GF2X_ELEMENT, x);
      /* ---------------------------------- */

      /* SCHOOLBOOK UPDATE OF THE SYNDROME */
      gf2x_mod_mul_monom(update, x == 0 ? 0 :  x, HTr[block]);
      gf2x_xor(privateSyndrome, update, privateSyndrome);
      //update_counters_after_flip(sigma, HtrPosOnes, flip, privateSyndrome);
      update_counters_bitsliced(bs_unsatParityChecks, HtrPosOnes, H_dense, privateSyndrome, flip);
      hw = population_count(privateSyndrome);
      /* ---------------------------------- */

      iter++;
   } while( (iter < 1.5*NUM_ERRORS_T) && (hw != 0) );


   /* Check the solution of the decoder */
   int check = 0;
   while (check < NUM_DIGITS_GF2X_ELEMENT && privateSyndrome[check++] == 0);
   return (check == NUM_DIGITS_GF2X_ELEMENT);
      
}
