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

static inline POSITION_T shift(POSITION_T i, POSITION_T h){
   POSITION_T pos = h + i;
   POSITION_T mask = -(pos >= P);
   return pos - (P & mask);
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
        if (bits) return i + __builtin_ctz(bits);
    }

    // residual positions
    for (i = (len / 32) * 32; i < len; i++) { if (arr[i] == max) return i; }

    return -1;
}

void compute_counters(const bs_operand_t* bs, uint8_t* ctrs, int total_elements, int bitsliced_width) {
   memset(ctrs, 0, total_elements * sizeof(uint8_t));

   for (int j = 0; j < total_elements; j++) {
       int block       = j / 256;  // quale blocco
       int lane        = (j / 64) % 4; // quale lane
       int bit_in_lane = j % 64; // quale posizione nella lane

       for (int i = 0; i < bitsliced_width; i++) {
           uint64_t lanes[4];
           memcpy(lanes, &bs[block].slice[i], sizeof(lanes));
           uint64_t bit = (lanes[lane] >> bit_in_lane) & 1ULL;
           ctrs[j] += (uint8_t)(bit << i);
       }
   }
}

void compute_counters_be(const bs_operand_t* bs, uint8_t* ctrs, int total_elements, int bitsliced_width) {

   memset(ctrs, 0, total_elements * sizeof(uint8_t));

   for (int j = 0; j < total_elements; j++) {
       int adjusted    = j + SLACK_SIZE;       // SLACK_SIZE = 61
       int block       = adjusted / 256;
       int lane        = (adjusted / 64) % 4;
       int bit_in_lane = 63 - (adjusted % 64); // big-endian dentro il DIGIT

       uint64_t lanes[4];
       for (int i = 0; i < bitsliced_width; i++) {
           memcpy(lanes, &bs[block].slice[i], sizeof(lanes));
           uint64_t bit = (lanes[lane] >> bit_in_lane) & 1ULL;
           ctrs[j] += (uint8_t)(bit << i);
       }
   }
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
      gf2x_mod_densify_CT(HTr[i],HtrPosOnes[i],V);
   }
   /* In this way we can update the counter as a xor between syndrome and h_i*/

   uint8_t sigma[N0*P] __attribute__((aligned(32)));
   int iter = 0;

   DIGIT tmp[NUM_DIGITS_GF2X_ELEMENT] = {0};
   DIGIT h[NUM_DIGITS_GF2X_ELEMENT] = {0};
   DIGIT update[NUM_DIGITS_GF2X_ELEMENT] = {0};
   int hw = population_count(privateSyndrome);

   /* COMPUTING COUNTERS BITSLICED */
   bs_operand_t bs_unsatParityChecks[N0*NUM_SLICES_GF2X_ELEMENT];
   memset(bs_unsatParityChecks, 0, sizeof(bs_unsatParityChecks));
   for (int i = 0; i < N0; i++) {
      lift_mul_dense_to_sparse_CT(bs_unsatParityChecks+(i*NUM_SLICES_GF2X_ELEMENT),
      privateSyndrome,
      HPosOnes[i],
      V);
   }
 
   /* FIND ARGMAX IN BITSLICED STRUCT */
   __m256i candidate[N0*NUM_SLICES_GF2X_ELEMENT] = {0};
   // set that every position is a possible candidate
   for(int z=0; z< N0*NUM_SLICES_GF2X_ELEMENT; z++) candidate[z] = _mm256_cmpeq_epi32(candidate[0], candidate[0]);

   /* SCAN BITSLICED COUNTER ARRAY FROM THE MSB TO THE LSB TO UPDATE THE CANDIDATE ARRAY */
   for(int i=BITSLICED_OPERAND_WIDTH-1; i >= 0; i--){

      uint32_t zero_ctr = 0;
      for(int z=0; z< N0*NUM_SLICES_GF2X_ELEMENT; z++){
         zero_ctr += _mm256_testz_si256(candidate[z], bs_unsatParityChecks[z].slice[i]);
      }
      
      if(i>3){
         printf("Zero ctr: %u in total %u \n", zero_ctr,N0*NUM_SLICES_GF2X_ELEMENT);
      }
      
      if(zero_ctr != N0*NUM_SLICES_GF2X_ELEMENT){
         for(int z=0; z< N0*NUM_SLICES_GF2X_ELEMENT; z++){
            candidate[z] = _mm256_and_si256(candidate[z], bs_unsatParityChecks[z].slice[i]);
         }
      }

   }

   int pos = 0;
   uint8_t found = 0;
   /* FROM THE CANDIDATE ARRAY FIND THE POSITION OF THE ARGMAX */
   for(int i =0; i < N0*NUM_SLICES_GF2X_ELEMENT; i++){
      
      if(_mm256_testz_si256(candidate[i], candidate[i])){
         pos += 256;
      }else{
         
         printf("Candidate is not all zeroes! \n");

         uint64_t x;

         pos += __builtin_ctzll(_mm256_extract_epi64(candidate[i], 0));
         printf("Pos is summed to %lu \n",__builtin_ctzll(_mm256_extract_epi64(candidate[i], 0)));
         if((pos & (0x3f)) != 0){
            printf("I'm breaking porcodio\n");
            found = 1;
            break;
         }
         pos += __builtin_ctzll(_mm256_extract_epi64(candidate[i], 1));
         if((pos & (0x3f)) != 0){
            found = 1;
            break;
         }
         pos += __builtin_ctzll(_mm256_extract_epi64(candidate[i], 2));
         if((pos & (0x3f)) != 0){
            found = 1;
            break;
         }
         pos += __builtin_ctzll(_mm256_extract_epi64(candidate[i], 3));
         if((pos & (0x3f)) != 0){
            found = 1;
            break;
         }
      }
   }

   

   /* CHECK IF THE ARGMAX FOUND IS THE SAME WITH A SCHOOLBOOK APPROACH */
   /*
   uint8_t ctrs[N0*P] = {0};
   
   for(int j=0; j<N0*P; j++){
      
      for(int i=0; i<BITSLICED_OPERAND_WIDTH;i++){
         

      uint64_t slice_slice;
        switch(j>>6){
         case 0: slice_slice = _mm256_extract_epi64(bs_unsatParityChecks[j >> 8].slice[i],0); break;
         case 1: slice_slice = _mm256_extract_epi64(bs_unsatParityChecks[j >> 8].slice[i],1); break;
         case 2: slice_slice = _mm256_extract_epi64(bs_unsatParityChecks[j >> 8].slice[i],2); break;
         case 3: slice_slice = _mm256_extract_epi64(bs_unsatParityChecks[j >> 8].slice[i],3); break;
      }
      uint64_t bit =  slice_slice & (1 << (j%64));
      bit =  bit >> ((j%64));
      ctrs[j] += bit << i;
   }
}
*/

   uint8_t ctrs[N0 * P];
   compute_counters_be(bs_unsatParityChecks, ctrs, N0 * P, BITSLICED_OPERAND_WIDTH);

   int max = 0;
   for (int i = 0; i < N0 * P; i++) {
      if (ctrs[i] > ctrs[max]) max = i;
   }



   
   for(int i=0; i<N0; i++){
      
      POSITION_T offset = i * P;
      memcpy(h, HTr[i], DIGIT_SIZE_B * NUM_DIGITS_GF2X_ELEMENT);

      for(int j=0; j < P; j++){
            // in teoria la riscrivo tutta quindi non c'è bisogno 
         gf2x_and(tmp, h, privateSyndrome);
         sigma[j + offset] = population_count(tmp);
         gf2x_mod_mul_monom(h,  1, h);
      }
   }
      
   POSITION_T flip_pos = argmax_avx2(sigma, N0*P);

   int flip_pos_2 = 0;
   for(int i = 0; i < N0*P; i++){
       if(sigma[i] > sigma[flip_pos_2]) flip_pos_2 = i;
   }

   printf("--------------------------------------\n");
   printf("ARGMAX position founded : \n");
   printf("\t ARGMAX from bitsliced:  \t %d \n", pos);
   printf("\t ARGMAX from extraction: \t %d \n", max);
   printf("\t ARGMAX from schoolbook avx:\t %d \n", flip_pos);
   printf("\t ARGMAX from schoolbook: \t %d \n", flip_pos_2);
   printf("--------------------------------------\n");



   
   do{
      /* Zeroed all variable  */
      //memset(sigma, 0, N0*P*sizeof(uint8_t));
      //memset(update, 0, DIGIT_SIZE_B*NUM_DIGITS_GF2X_ELEMENT);
      
      
      /* Compute counters dense */
      /* In gf2x_and possiamo evitare di considerare l'endianess e farlo diretto tanto usano la stessa rappresentazione */
      for(int i=0; i<N0; i++){
         POSITION_T offset = i * P;
         memcpy(h, HTr[i], DIGIT_SIZE_B * NUM_DIGITS_GF2X_ELEMENT);

         for(int j=0; j < P; j++){
            // in teoria la riscrivo tutta quindi non c'è bisogno 
            gf2x_and(tmp, h, privateSyndrome);
            sigma[j + offset] = population_count(tmp);
            gf2x_mod_mul_monom(h,  1, h);
         }
      }
      
      POSITION_T flip = argmax_avx2(sigma, N0*P);
      int block    = flip / P;  // quale blocco di HTr
      int x        = flip % P;  // di quanto ruotare dentro quel blocco
      gf2x_toggle_coeff(out + block * NUM_DIGITS_GF2X_ELEMENT, x);
      
      gf2x_mod_mul_monom(update, x == 0 ? 0 :  x, HTr[block]);
      gf2x_xor(privateSyndrome, update, privateSyndrome);
      hw = population_count(privateSyndrome);
      iter++;
   } while( (iter < 2*NUM_ERRORS_T) && (hw != 0) );

   /* Check the solution of the decoder */
   int check = 0;
   while (check < NUM_DIGITS_GF2X_ELEMENT && privateSyndrome[check++] == 0);
   return (check == NUM_DIGITS_GF2X_ELEMENT);
      
}
