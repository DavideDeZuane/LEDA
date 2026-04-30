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

static inline void shift_positions(const POSITION_T *in, POSITION_T *out, int n, POSITION_T shift){
    
    __m256i vp     = _mm256_set1_epi32((uint32_t)P);
    __m256i vshift = _mm256_set1_epi32((uint32_t)shift);

    int r = 0;
    for (; r <= n - 8; r += 8) {
        __m256i h   = _mm256_loadu_si256((__m256i *)(in + r));
        __m256i sum = _mm256_add_epi32(h, vshift);
        __m256i sub = _mm256_sub_epi32(sum, vp);
        __m256i msk = _mm256_cmpgt_epi32(vp, sum);
        __m256i res = _mm256_blendv_epi8(sub, sum, msk);
        _mm256_storeu_si256((__m256i *)(out + r), res);
    }
    // residui
    for (; r < n; r++) {
        out[r] = in[r] + shift;
        if (out[r] >= P) out[r] -= P;
    }
}
/* ------------------------------------------------------------------------------------------*/

/*#########################################################################################*/
/*FUNCTIONS FOR UINT8 COUNTER ARRAY STRUCTURE                                              */
/*#########################################################################################*/

POSITION_T argmax_uint8(const uint8_t* arr, size_t len) {

    /* ------------------------------------------------- */
    /*  Find Max                                         */
    /* ------------------------------------------------- */
    size_t i = 0;
    __m512i max_vec = _mm512_setzero_si512();
    for (; i <= len - 64; i += 64) {
        __m512i v = _mm512_loadu_si512((__m512i*)&arr[i]);
        max_vec = _mm512_max_epu8(max_vec, v);
    }
    // residui con AVX2
    __m256i max256 = _mm256_max_epu8(
                         _mm512_extracti64x4_epi64(max_vec, 0),
                         _mm512_extracti64x4_epi64(max_vec, 1));
    for (; i <= len - 32; i += 32) {
        __m256i v = _mm256_loadu_si256((__m256i*)&arr[i]);
        max256 = _mm256_max_epu8(max256, v);
    }

    // riduzione orizzontale AVX2
    __m128i lo = _mm256_castsi256_si128(max256);
    __m128i hi = _mm256_extracti128_si256(max256, 1);
    __m128i m  = _mm_max_epu8(lo, hi);
    m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
    uint8_t max_val = (uint8_t)_mm_extract_epi8(m, 0);

    // residui scalari
    for (; i < len; i++)
        if (arr[i] > max_val) max_val = arr[i];

    /* ------------------------------------------------- */
    /*  Find Argmax                                      */
    /* ------------------------------------------------- */
    __m512i vmax = _mm512_set1_epi8((char)max_val);

    for (i = 0; i <= len - 64; i += 64) {
        __m512i v    = _mm512_loadu_si512((__m512i*)&arr[i]);
        __mmask64 msk = _mm512_cmpeq_epi8_mask(v, vmax);
        if (msk) return (POSITION_T)(i + __builtin_ctzll(msk));
    }
    for (; i <= len - 32; i += 32) {
        __m256i v    = _mm256_loadu_si256((__m256i*)&arr[i]);
        __m256i vmax256 = _mm256_set1_epi8((char)max_val);
        __m256i cmp  = _mm256_cmpeq_epi8(v, vmax256);
        int bits = _mm256_movemask_epi8(cmp);
        if (bits) return (POSITION_T)(i + __builtin_ctz(bits));
    }
    for (; i < len; i++)
        if (arr[i] == max_val) return (POSITION_T)i;

    return (POSITION_T)-1;
}

static inline void compute_counters_uint8(uint8_t *sigma, const POSITION_T H[2][V], const DIGIT *syndrome){

    const int first_w = NUM_DIGITS_GF2X_ELEMENT - 1 - (P / DIGIT_SIZE_b);

    // Converti H in int32 per i load AVX2
    int32_t h32[2][V];
    for (int b = 0; b < 2; b++)
        for (int j = 0; j < V; j++)
            h32[b][j] = (int32_t)H[b][j];

    for (int b = 0; b < 2; b++) {
        const int32_t *h  = h32[b];
        uint8_t       *sig = sigma + b * P;

        for (int w = first_w; w < NUM_DIGITS_GF2X_ELEMENT; w++) {
            DIGIT word = syndrome[w];
            if (w == first_w) word &= SLACK_CLEAR_MASK;
            if (word == 0) continue;

            while (word) {
                int bit      = __builtin_clzll(word);
                int poly_idx = w * 64 + bit - SLACK_SIZE;

                if (poly_idx >= P) break;

                int32_t p  = (int32_t)((P - 1) - poly_idx);
                __m256i vp = _mm256_set1_epi32(p);
                __m256i vP = _mm256_set1_epi32((int32_t)P);

                int j = 0;
                for (; j <= V - 8; j += 8) {
                    __m256i vh       = _mm256_loadu_si256((const __m256i *)(h + j));
                    __m256i diff     = _mm256_sub_epi32(vp, vh);
                    __m256i adj      = _mm256_add_epi32(diff, vP);
                    __m256i neg_mask = _mm256_cmpgt_epi32(vh, vp);
                    __m256i ell_vec  = _mm256_blendv_epi8(diff, adj, neg_mask);

                    int32_t ells[8];
                    _mm256_storeu_si256((__m256i *)ells, ell_vec);

                    sig[ells[0]]++;
                    sig[ells[1]]++;
                    sig[ells[2]]++;
                    sig[ells[3]]++;
                    sig[ells[4]]++;
                    sig[ells[5]]++;
                    sig[ells[6]]++;
                    sig[ells[7]]++;
                }
                // coda scalare se V non è multiplo di 8
                for (; j < V; j++) {
                    int32_t diff = p - h[j];
                    sig[diff >= 0 ? diff : diff + (int32_t)P]++;
                }

                word &= ~(1ULL << (63 - bit));
            }
        }    
    }
}

static inline void update_counters_uint8(uint8_t *sigma, const POSITION_T HtrPosOnes[N0][V], const POSITION_T  HPosOnes[N0][V], POSITION_T pos_flip, DIGIT* syndrome){

    int b = pos_flip >= P ? 1 : 0;
    POSITION_T local_pos = pos_flip - b * P;

    __m256i vp   = _mm256_set1_epi32((uint32_t)P);
    __m256i vpos = _mm256_set1_epi32((uint32_t)local_pos);

    // pre-carica HPosOnes[b2] nei registri AVX2 una volta sola
    __m256i h2_regs[N0][N_REGS];
    for (int b2 = 0; b2 < N0; b2++) {
        for (int r = 0; r < N_REGS; r++) {
            uint32_t tmp[8] = {0};
            for (int j = 0; j < 8 && r*8+j < V; j++)
                tmp[j] = HPosOnes[b2][r*8+j];
            h2_regs[b2][r] = _mm256_loadu_si256((__m256i *)tmp);
        }
    }

    // calcola row_indices, ds e aggiorna counter in un unico loop
    for (int r = 0; r < N_REGS; r++) {
        uint32_t tmp[8] = {0};
        for (int i = 0; i < 8 && r*8+i < V; i++)
            tmp[i] = HtrPosOnes[b][r*8+i];
        __m256i htr = _mm256_loadu_si256((__m256i *)tmp);
        __m256i sum = _mm256_add_epi32(htr, vpos);
        __m256i sub = _mm256_sub_epi32(sum, vp);
        __m256i msk = _mm256_cmpgt_epi32(vp, sum);
        __m256i res = _mm256_blendv_epi8(sub, sum, msk);
        _mm256_storeu_si256((__m256i *)tmp, res);

        for (int i = 0; i < 8 && r*8+i < V; i++) {
            POSITION_T row_index = tmp[i];

            // ds branch-free
            int straightIdx = (NUM_DIGITS_GF2X_ELEMENT * DIGIT_SIZE_b - 1) - row_index;
            DIGIT bit       = (syndrome[straightIdx / DIGIT_SIZE_b] >>
                              (DIGIT_SIZE_b - 1 - straightIdx % DIGIT_SIZE_b)) & 1;
            int d           = (int)(2 * bit) - 1;

            __m256i vrow = _mm256_set1_epi32((uint32_t)row_index);

            for (int b2 = 0; b2 < N0; b2++) {
                POSITION_T offset = b2 * P;

                for (int r2 = 0; r2 < N_REGS; r2++) {
                    // col = (HPosOnes[b2][j] + row_index) % P
                    __m256i col = _mm256_add_epi32(h2_regs[b2][r2], vrow);
                    __m256i s   = _mm256_sub_epi32(col, vp);
                    __m256i m   = _mm256_cmpgt_epi32(vp, col);
                    col = _mm256_blendv_epi8(s, col, m);

                    uint32_t cols[8];
                    _mm256_storeu_si256((__m256i *)cols, col);

                    for (int j = 0; j < 8 && r2*8+j < V; j++)
                        sigma[offset + cols[j]] += d;
                }
            }
        }
    }

    /*

    // calcola quanto pè la variazione del counter per ogni riga
    int ds[V];
    for (int i = 0; i < V; i++)
        ds[i] = gf2x_get_coeff(syndrome, row_indices[i]) ? 1 : -1;

   // aggiorna i counter usando HPosOnes
   for (int i = 0; i < V; i++) {
    POSITION_T row_index = row_indices[i];
    int d = ds[i];

    for (int b2 = 0; b2 < N0; b2++) {
        POSITION_T offset = b2 * P;
        for (int j = 0; j < V; j++) {
            POSITION_T ell = row_index + HPosOnes[b2][j] ;
            if (ell >= P) ell -= P;
            sigma[offset + ell] += d;
        }
    }
}

    #undef N_REGS
    
    */
}
/* ------------------------------------------------------------------------------------------*/

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

static inline int argmax_bitsliced_impv(const bs_operand_t* bs, int n_slices_total) {

    //__m256i *candidate = aligned_alloc(32, N0*NUM_SLICES_GF2X_ELEMENT * sizeof(__m256i));
   __m256i candidate[N0*NUM_SLICES_GF2X_ELEMENT] __attribute__((aligned(32)));
   for(int z = 0; z < N0*NUM_SLICES_GF2X_ELEMENT; z++)
       candidate[z] = _mm256_cmpeq_epi32(candidate[z], candidate[z]);

   /* phase 1: scan MSB→LSB with early exit */
   int active_blocks = N0*NUM_SLICES_GF2X_ELEMENT;
   for(int i = BITSLICED_OPERAND_WIDTH-1; i >= 0; i--){
        // move this outside the loop
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

static inline void update_counters_bitsliced(
    bs_operand_t     bs_unsatParityChecks[N0*NUM_SLICES_GF2X_ELEMENT],
    const POSITION_T HtrPosOnes[N0][V],
    const POSITION_T HPosOnes[N0][V],
    const DIGIT      syndrome[],
    POSITION_T       pos_flip
) {


/*
int b = pos_flip >= P ? 1 : 0;
POSITION_T local_pos = pos_flip - b * P;

        POSITION_T row_indices[V];

        shift_positions(HtrPosOnes[b], row_indices, V, local_pos);
        
        // leggi i segni dalla sindrome per tutti i row_index
        int ds[V];
        for (int i = 0; i < V; i++)
            ds[i] = gf2x_get_coeff(syndrome, row_indices[i]) ? 1 : -1;
    
            // aggiorna i counter: separa inc e dec per minimizzare le chiamate
            SLICE_TYPE tmp_slice_2[N0*NUM_SLICES_GF2X_ELEMENT]; //piu 2
            SLICE_TYPE tmp_slice_1[N0*NUM_SLICES_GF2X_ELEMENT]; //piu 1
            SLICE_TYPE tmp_slice_m1[N0*NUM_SLICES_GF2X_ELEMENT]; //meno 1
        SLICE_TYPE tmp_slice_m2[N0*NUM_SLICES_GF2X_ELEMENT]; //meno 2

        POSITION_T update[V];
        
        for (int b2 = 0; b2 < N0; b2++) {
            POSITION_T offset = b2*P;
            for(int i = 0; i <V; i++){
                shift_positions(HPosOnes[b2], update, V, row_indices[i]);
                int d = ds[i];
                for(int j=0; j < V; j++){
                    POSITION_T to_update = update[j];
                    switch(d){
                        case 1: 
                            // tocca cambaire il toggle coeff perche' quello lavora a blocchi di 64 mentre su avx abbiamo blocchi da 256 
                            // quindi tocca scrivere una funzione custom per fare queta cosa
                            // le posizioni qui vanno aggiornate con l'offset del blocco che stiamo considerando
                            // inoltre qua i controlli sono fatti male, vanno scritti meglio 
                            if(gf2x_get_coeff((DIGIT *)tmp_slice_1, to_update) ==1) gf2x_toggle_coeff((DIGIT *)tmp_slice_2, to_update);
                            else gf2x_toggle_coeff((DIGIT *)tmp_slice_1, to_update);
                            break;
                            case 0: 
                            if(gf2x_get_coeff((DIGIT *)tmp_slice_m1, to_update) ==1) gf2x_toggle_coeff((DIGIT *)tmp_slice_m2, to_update);
                            else gf2x_toggle_coeff((DIGIT *)tmp_slice_m1, to_update);
                            break;
                            
                            default:
                            break;
                        }
                        
                    }
                    
                }
                
                
            }
            
            
            for (int j = 0; j < N0*NUM_SLICES_GF2X_ELEMENT; j++){
                
            bs_operand_t *bs_block = bs_unsatParityChecks;
            
            bs_block[j] = bitslice_inc(bs_block[j], tmp_slice_1[j]);
            bs_block[j] = bitslice_inc(bs_block[j], tmp_slice_2[j]);
            
            bs_block[j] = bitslice_dec(bs_block[j], tmp_slice_m1[j]);
            bs_block[j] = bitslice_dec(bs_block[j], tmp_slice_m2[j]);
        }
        */
    
}
/* ------------------------------------------------------------------------------------------*/

/*#########################################################################################*/
/* CONVERSION FUNCTION FROM SLICED -> UINT8                                                */
/*#########################################################################################*/

void sliced_to_uint8(const bs_operand_t* bs, uint8_t* ctrs, int total_elements, int bitsliced_width) {
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
/* ------------------------------------------------------------------------------------------*/


/*#########################################################################################*/
/* DECODER                                                                                 */
/*#########################################################################################*/
int bf_decoding_CT(DIGIT out[], const POSITION_T HtrPosOnes[N0][V], const POSITION_T HPosOnes[N0][V], DIGIT privateSyndrome[]){

    /* Densify HTr */
    DIGIT HTr[N0][NUM_DIGITS_GF2X_ELEMENT] = {{0}};
    for(int i=0; i<N0; i++) {
        gf2x_mod_densify_VT(HTr[i],HtrPosOnes[i],V);
    }

    int iter = 0;
    int hw = population_count(privateSyndrome);
    
    DIGIT update[NUM_DIGITS_GF2X_ELEMENT];
    bs_operand_t bs_unsatParityChecks[N0*NUM_SLICES_GF2X_ELEMENT];
    memset(bs_unsatParityChecks, 0, sizeof(bs_unsatParityChecks));
    
    for (int i = 0; i < N0; i++) {
        lift_mul_dense_to_sparse_CT(
            bs_unsatParityChecks+(i*NUM_SLICES_GF2X_ELEMENT),
            privateSyndrome,
            HPosOnes[i],
            V
        );
    }

    uint8_t sigma[N0*P] __attribute__((aligned(32)));
    memset(sigma, 0, N0*P*sizeof(uint8_t));
    /* CONVERSION OF THE COUNTERS */
    sliced_to_uint8(bs_unsatParityChecks, sigma, N0*P, BITSLICED_OPERAND_WIDTH);

    //compute_counters_uint8(sigma, HtrPosOnes, privateSyndrome);


   do{
        memset(update, 0, NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_B);
        /* HYBRID APPROACH WITH COUNTER ARRAY FROM SLICED TO UINT8 */
        POSITION_T flip = argmax_uint8(sigma, N0*P);
        int block    = flip / P;  // quale blocco di HTr
        int x        = flip % P;  // di quanto ruotare dentro quel blocco
        gf2x_toggle_coeff(out + block * NUM_DIGITS_GF2X_ELEMENT, x);
        gf2x_mod_mul_monom(update, x == 0 ? 0 :  x, HTr[block]);
        gf2x_xor(privateSyndrome, update, privateSyndrome);
        update_counters_uint8(sigma, HtrPosOnes, HPosOnes, flip, privateSyndrome);
        
        /* APPROACH WITH COUNTER ARRAY BITSLICED */
        //POSITION_T flip = argmax_bitsliced_impv(bs_unsatParityChecks, N0 * NUM_SLICES_GF2X_ELEMENT);
        //int block    = flip / P;  // quale blocco di HTr
        //int x        = flip % P;  // di quanto ruotare dentro quel blocco
        //gf2x_toggle_coeff(out + block * NUM_DIGITS_GF2X_ELEMENT, x);
        //update_counters_bitsliced(bs_unsatParityChecks, HtrPosOnes, HPosOnes, privateSyndrome, flip);

        hw = population_count(privateSyndrome);
        iter++;

   } while( (iter < 1.5*NUM_ERRORS_T) && (hw != 0) );


   /* Check the solution of the decoder */
   int check = 0;
   while (check < NUM_DIGITS_GF2X_ELEMENT && privateSyndrome[check++] == 0);
   //return (check == NUM_DIGITS_GF2X_ELEMENT);
   return 1;
      
}
