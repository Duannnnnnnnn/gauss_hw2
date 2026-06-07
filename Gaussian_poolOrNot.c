/*
 *  Author: Paul Horton
 *  Copyright: Paul Horton 2021, All rights reserved.
 *  Created: 20211201
 *  Updated: 20260513
 *  Licence: GPLv3
 *  Description: Simple demonstration of a Bayesian way to guess at the number of components
 *               behind a sample of numerical data.
 *  Compile:  gcc -Wall -O3 -o Gaussian_poolOrNot Gaussian_poolOrNot.c GSLfun.c timespec_utils.c -lgsl -lgslcblas -lm
 *  Environment: $GSL_RNG_SEED
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "GSLfun.h"
#include "timespec_utils.h"
/* ───────────  Global definitions and variables  ────────── */
#define DATA_N 50
#define CDF_GAUSS_N 20
#define CDF_GAMMA_N 10
#define CDF_JBETA_N 40

// Block of code used twice in main()
#define COMPUTE_DATA_PROBS_PRINT_RESULTS(NUM)\
  prob_data1_bySampling=  data_prob_1component_bySampling();               \
  prob_data2_bySampling=  data_prob_2component_bySampling();               \
  prob_data1_bySumming =  data_prob_1component_bySumming();                \
  prob_data2_bySumming =  data_prob_2component_bySumming();                \
  if(  prob_data1_bySampling > prob_data2_bySampling  )   ++model##NUM##_sampling_favors1; \
  if(  prob_data1_bySumming  > prob_data2_bySumming   )   ++model##NUM##_summing__favors1; \
  printf( "Data maximum likelihood under (one,two) component model= (%g,%g)\n", \
          data_Gauss1_maxLikelihood(), data_Gauss2_maxLikelihood()        \
          );                                                               \
  printf( "Integrals by sampling= (%g,%g)  by summing: (%g,%g)\n\n",       \
          prob_data1_bySampling, prob_data2_bySampling,                    \
          prob_data1_bySumming, prob_data2_bySumming )



#define ADD_TIME_AFTER_BEFORE(METHOD) \
  timespec_setToNow( &after_time );      \
  tot_msec_time[METHOD]  += timespec_diff_msecs( before_time, after_time )


typedef struct{
  double mixCof;
  Gauss_params Gauss1;
  Gauss_params Gauss2;
} Gauss_mixture_params;


const Gauss_params mu_prior_params= {0.0, 4.0};
const double sigma_prior_param_a= 0.5;
const double sigma_prior_param_b= 2.0;


double data[DATA_N];
const uint dataN= DATA_N;

const uint sampleRepeatNum= 2000000;


// To record computation times in milliseconds for each step or method you wish to record.
double tot_msec_time[5];

// Names for positions in array `tot_msec_time'.
// "T_" prefix used here mainly to reduce the risk of name conflicts.
enum {SUM_PRECOMP, T_SUM1, T_SAMP1, T_SUM2, T_SAMP2};


// Vars (re)used by functions below to hold times.
struct timespec before_time;
struct timespec after_time;



/* ───────────  Functions to help summarize or dump the data  ────────── */

double data_sample_mean(){
  double mean= 0.0;
  for( uint i= 0;  i < dataN;  ++i ){
    mean += data[i];
  }
  return  mean / (double) dataN;
}

double data_sample_variance(){
  double mean= data_sample_mean();
  double var= 0.0;
  for( uint i= 0;  i < dataN;  ++i ){
    double diff=  data[i] - mean;
    var +=  diff * diff;
  }
  return  var / (double) (dataN);
}



// ternary CMP function for use with qsort
int CMPdata( const void *arg1, const void *arg2 ){
  return(
         (*(double*) arg1 < *(double*) arg2)? -1 :
         (*(double*) arg2 < *(double*) arg1)? +1 :
         /* else    *arg1 == *arg2   */        0);
}

void data_print(){
  qsort(  data,  dataN,  sizeof(double), CMPdata  );
  for( uint i= 0;  i < dataN;  ++i ){
    printf( "%u\t+%5.3f ", i, data[i] );
  }
}


/* ───────────  Functions used for sampling/generating data   ────────── */

Gauss_params prior_Gauss_params_sample(){
  Gauss_params params;
  params.mu=   GSLfun_ran_gaussian( mu_prior_params );
  params.sigma=  sigma_of_precision( GSLfun_ran_gamma(sigma_prior_param_a, sigma_prior_param_b) );
  return  params;
}


Gauss_mixture_params prior_Gauss_mixture_params_sample(){

  // You can adjust these parameters, but cdfInv_precompute assumes a=b
  static const double betaDist_a= 0.75;
  static const double betaDist_b= 0.75;

  Gauss_mixture_params params;
  params.mixCof=  GSLfun_ran_beta( betaDist_a, betaDist_b );
  params.Gauss1=  prior_Gauss_params_sample();
  params.Gauss2=  prior_Gauss_params_sample();
  return  params;
}


void data_generate_1component( Gauss_params params ){
  for( uint i= 0; i < dataN; ++i ){
    data[i]=  GSLfun_ran_gaussian( params );
  }
}

void data_generate_2component( Gauss_mixture_params params ){
  for( uint i= 0; i < dataN; ++i ){
    data[i]=  GSLfun_ran_gaussian
      (gsl_ran_flat01() < params.mixCof?  params.Gauss1  : params.Gauss2);
  }
}


/* ───────────  Numerical integration precomputation  ────────── */

// Arrays to hold precomputed values.
double cdfInv_Gauss[CDF_GAUSS_N];  const double cdf_Gauss_n= CDF_GAUSS_N;
double cdfInv_gamma[CDF_GAMMA_N];  const double cdf_gamma_n= CDF_GAMMA_N;
double cdfInv_JBeta[CDF_JBETA_N];  const double cdf_JBeta_n= CDF_JBETA_N;

//  Precompute the cumulative probabilities of μ and σ discrete values.
//  The probabilities depend on the current prior_params values
void cdfInv_precompute(){
  double x;

  timespec_setToNow( &before_time );

  // Since Normal range is unbounded, precompute cdfInv for vals:  ¹⁄₍ₙ₊₁₎...ⁿ⁄₍ₙ₊₁₎
  for(  uint i= 0; i < cdf_Gauss_n; ++i  ){
    x= (i+1) / (double) (1+cdf_Gauss_n);
    cdfInv_Gauss[i]=  gsl_cdf_gaussian_Pinv( x, mu_prior_params.sigma );
  }
  for(  uint i= 0; i < cdf_gamma_n; ++i  ){
    x= i / (double) (cdf_gamma_n);
    cdfInv_gamma[i]=  gsl_cdf_gamma_Pinv( x, sigma_prior_param_a, sigma_prior_param_b );
    //printf( "cdfInv_Gamma[%u]= %g\n", i, cdfInv_gamma[i] );
  }
  for(  uint i= 0; i < cdf_JBeta_n; ++i  ){
    // By symmetry, only need Beta values for p ≦ 0.5.  For example p=0.8, is the same p=0.2 with Gauss components swapped.
    x= 0.5 * i / (double) (cdf_JBeta_n);
    cdfInv_JBeta[i]=  gsl_cdf_beta_Pinv( x, 0.5, 0.5 );
    //printf( "cdfInv_JBeta[%u]= %g\n", i, cdfInv_JBeta[i] );
  }

  ADD_TIME_AFTER_BEFORE( SUM_PRECOMP );
}




/* ───────────  Probability Computations on the Data  ────────── */

// Return Ｐ[D|μ,σ]
double prob_data_given_1Gauss( const Gauss_params params ){
  double prob= 1.0;
  for(  uint d= 0;  d < dataN;  ++d  ){
    prob *= GSLfun_ran_gaussian_pdf( data[d], params );
  }
  return prob;
}


// Return Ｐ[D|m,μ₁,σ₁,μ₂,σ₂]
double prob_data_given_2Gauss( const double mixCof, const Gauss_params Gauss1, const Gauss_params Gauss2  ){
    double prob= 1.0;
    for( uint i= 0; i < dataN; ++i ){
      prob *=   (1-mixCof) * GSLfun_ran_gaussian_pdf( data[i], Gauss2 )
              +    mixCof  * GSLfun_ran_gaussian_pdf( data[i], Gauss1 );
    }
    return prob;
}


/* Return maximum likelihood of the data using a single Gaussian
 *
 * max_{μ,σ} Ｐ[D|μ,σ}
*/
double data_Gauss1_maxLikelihood(){
  Gauss_params params=  { data_sample_mean(), sqrt( data_sample_variance() ) };
  return prob_data_given_1Gauss( params );
}


double data_Gauss2_maxLikelihood(){
  // 初始化：用資料的均值±標準差作為兩個高斯的起始點
  double mean = data_sample_mean();
  double stdv = sqrt( data_sample_variance() );

  double mu1 = mean - 0.5*stdv,  sigma1 = stdv;
  double mu2 = mean + 0.5*stdv,  sigma2 = stdv;
  double mixCof = 0.5;  // b：屬於高斯2的比例

  double r[DATA_N];  // r[i] = 點i屬於高斯1的機率

  for( int iter = 0; iter < 200; ++iter ){
    // E步：算每個點屬於高斯1的機率
    for( uint i = 0; i < dataN; ++i ){
      Gauss_params p1 = {mu1, sigma1};
      Gauss_params p2 = {mu2, sigma2};
      double w1 = (1-mixCof) * GSLfun_ran_gaussian_pdf( data[i], p1 );
      double w2 =    mixCof  * GSLfun_ran_gaussian_pdf( data[i], p2 );
      r[i] = w1 / (w1 + w2);
    }

    // M步：用加權平均更新參數
    double sum_r = 0.0, sum_rx = 0.0, sum_rx2 = 0.0;
    double sum_s = 0.0, sum_sx = 0.0, sum_sx2 = 0.0;
    for( uint i = 0; i < dataN; ++i ){
      sum_r  +=  r[i];
      sum_rx +=  r[i] * data[i];
      sum_s  +=  (1-r[i]);
      sum_sx +=  (1-r[i]) * data[i];
    }
    mu1 = sum_rx / sum_r;
    mu2 = sum_sx / sum_s;
    for( uint i = 0; i < dataN; ++i ){
      double d1 = data[i] - mu1;
      double d2 = data[i] - mu2;
      sum_rx2 += r[i]     * d1*d1;
      sum_sx2 += (1-r[i]) * d2*d2;
    }
    sigma1  = sqrt( sum_rx2 / sum_r );
    sigma2  = sqrt( sum_sx2 / sum_s );
    mixCof  = sum_s / (double) dataN;

    // 避免 sigma 變成 0
    if( sigma1 < 1e-10 ) sigma1 = 1e-10;
    if( sigma2 < 1e-10 ) sigma2 = 1e-10;
  }

  Gauss_params p1 = {mu1, sigma1};
  Gauss_params p2 = {mu2, sigma2};
  return prob_data_given_2Gauss( mixCof, p1, p2 );
}



/* Compute Riemann sum to approximate the integral
 *
 * ∫ μ,σ  Ｐ[D|μ,σ]
 *
*/
double data_prob_1component_bySumming(){
  double prob_total= 0.0;

  timespec_setToNow( &before_time );
  for(  uint m= 0;  m < cdf_Gauss_n;  ++m  ){
    double mu= cdfInv_Gauss[m];
    for(  uint s= 0;  s < cdf_gamma_n;  ++s  ){
      double sigma=  sigma_of_precision( cdfInv_gamma[s] );
      Gauss_params cur_params= {mu, sigma};
      prob_total += prob_data_given_1Gauss( cur_params );
    }
  }

  ADD_TIME_AFTER_BEFORE( T_SUM1 );
  return  prob_total / (double) (cdf_Gauss_n * cdf_gamma_n);
}


/* Compute Riemann sum to approximate integral
 *
 * ∫ m,μ₁,σ₁,μ₂,σ₂  P[D|m,μ₁,σ₁,μ₂,σ₂]
 *
*/
double data_prob_2component_bySumming(){
  double prob_total= 0.0;

  timespec_setToNow( &before_time );
  for(  uint m1= 0;  m1 < cdf_Gauss_n;  ++m1  ){
    double mu1= cdfInv_Gauss[m1];
    for(  uint m2= 0;  m2 < cdf_Gauss_n;  ++m2  ){
      double mu2= cdfInv_Gauss[m2];
      for(  uint s1= 0;  s1 < cdf_gamma_n;  ++s1  ){
        double sigma1=  sigma_of_precision( cdfInv_gamma[s1] );
        Gauss_params cur_params1= {mu1, sigma1};
        for(  uint s2= 0;  s2 < cdf_gamma_n;  ++s2  ){
          double sigma2=  sigma_of_precision( cdfInv_gamma[s2] );
          Gauss_params cur_params2= {mu2, sigma2};
          for(  uint mi= 0;  mi < cdf_JBeta_n;  ++mi  ){
            double mixCof= cdfInv_JBeta[mi];
            prob_total +=  prob_data_given_2Gauss( mixCof, cur_params1, cur_params2 );
          }
        }
      }
    }
  }

  ADD_TIME_AFTER_BEFORE( T_SUM2 );
  return  prob_total / (double) (cdf_Gauss_n * cdf_Gauss_n * cdf_gamma_n * cdf_gamma_n * cdf_JBeta_n);
}



/*  Use sampling to estimate
 *  ∫ μ,σ  P[D|μ,σ]
 */
double data_prob_1component_bySampling(){
  double prob_total= 0.0;

  timespec_setToNow( &before_time );
  for( uint iter= 0;  iter < sampleRepeatNum; ++iter ){
    Gauss_params params= prior_Gauss_params_sample();
    prob_total += prob_data_given_1Gauss( params );

    // 每隔 10000 次印出當前估計值，用於觀察收斂
    if( (iter+1) % 10000 == 0 ){
      fprintf( stderr, "CONVERGE_M1\t%u\t%g\n", iter+1, prob_total/(iter+1) );
    }
  }

  ADD_TIME_AFTER_BEFORE( T_SAMP1 );
  return  prob_total / (double) sampleRepeatNum;
}



/*  Use sampling to estimate
 *  ∫ m,μ₁,σ₁,μ₂,σ₂  P[D|m,μ₁,σ₁,μ₂,σ₂]
 */
double data_prob_2component_bySampling(){
  double prob_total= 0.0;

  timespec_setToNow( &before_time );
  for( uint iter= 0;  iter < sampleRepeatNum; ++iter ){
    Gauss_mixture_params params=  prior_Gauss_mixture_params_sample();
    prob_total += prob_data_given_2Gauss( params.mixCof, params.Gauss1, params.Gauss2 );

    // 每隔 10000 次印出當前估計值，用於觀察收斂
    if( (iter+1) % 10000 == 0 ){
      fprintf( stderr, "CONVERGE_M2\t%u\t%g\n", iter+1, prob_total/(iter+1) );
    }
  }

  ADD_TIME_AFTER_BEFORE( T_SAMP2 );
  return  prob_total / (double) sampleRepeatNum;
}




// Is flag NAME given on command line?
// NAME includes leading hyphens, e.g. '--help'
bool cliFlagGivenP( int* argcR, char *argv[], const char* name ){
  for( int i= 0; i < *argcR; ++i ){
    // If NAME found splice it out and decrement ARGC
    if( !strcmp( argv[i], name ) ){
      for( int j= i; j < *argcR-1; ++j ){
         argv[j]= argv[j+1];
      }
      --(*argcR);
      return true;
    }
  }
  return false;
}


int main( int argc, char *argv[] ){

  uint datasets_n= 10;

  // So we can skip the 1-component generated datasets when we want to.
  bool skip_1component_datasetsP= cliFlagGivenP( &argc, argv, "--skip1c" );

  {
    char usage_fmt[]=  "Usage: %s [--skip1c] [num_datasets]\n";
    switch( argc ){
    case 1:
      break;
    case 2:
      datasets_n=  atoi( argv[1] );
      if( !datasets_n ){
        printf(  usage_fmt, argv[0]  );
        exit( 64 );
      }
      break;
    default:
      printf(  usage_fmt, argv[0]  );
      exit( 64 );
    }
  }

  GSLfun_setup();
  double prob_data1_bySampling, prob_data2_bySampling;
  double prob_data1_bySumming,  prob_data2_bySumming;

  cdfInv_precompute();


  uint model1_sampling_favors1=  0;
  uint model1_summing__favors1=  0;
  uint model2_sampling_favors1=  0;
  uint model2_summing__favors1=  0;



  printf(  "Starting computation for %d datasets each. ...\n",  datasets_n  );

  if( !skip_1component_datasetsP ){
    printf( "\nData generated with one component\n" );
    for(  uint iter= 0;  iter < datasets_n;  ++iter  ){
      Gauss_params model_params = prior_Gauss_params_sample();
      printf(  "generating data with: (μ,σ) =  (%4.2f,%4.2f)\n", model_params.mu, model_params.sigma  );
      data_generate_1component( model_params );
      COMPUTE_DATA_PROBS_PRINT_RESULTS(1);
    }
  }

  printf( "\nData generated with two components\n" );
  for(  uint iter= 0;  iter < datasets_n;  ++iter  ){
    Gauss_mixture_params model_params=  prior_Gauss_mixture_params_sample();
    printf(  "generating data with:  m; (μ1,σ1); (μ2,σ2) =  %5.3f; (%4.2f,%4.2f); (%4.2f,%4.2f)\n",
             model_params.mixCof,
             model_params.Gauss1.mu, model_params.Gauss1.sigma,
             model_params.Gauss2.mu, model_params.Gauss2.sigma  );
    data_generate_2component( model_params );
    COMPUTE_DATA_PROBS_PRINT_RESULTS(2);
  }

  printf( "By sampling: Model1 data, correct selection %u/%u\n", model1_sampling_favors1, datasets_n  );
  printf( "             Model2 data, correct selection %u/%u\n", (datasets_n - model2_sampling_favors1), datasets_n  );
  printf( "By summing:  Model1 data, correct selection %u/%u\n", model1_summing__favors1, datasets_n  );
  printf( "             Model2 data, correct selection %u/%u\n", (datasets_n - model2_summing__favors1), datasets_n  );

  printf( "total times spent are:\n-PRECOMP-\t--SUM1---\t--SAMP1--\t--SUM2---\t--SAMP2-\n" );
  for( int i= 0; i < sizeof(tot_msec_time)/sizeof(tot_msec_time[0]); i++ ){
    if( i )  printf( "\t" );
    printf( "%9g", tot_msec_time[i] );
  }
  printf( "\n" );
}
