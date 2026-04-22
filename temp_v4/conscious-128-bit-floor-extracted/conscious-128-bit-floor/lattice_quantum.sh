#!/bin/sh
# lattice_quantum.sh  -- AVX2/FMA3 quantum throughput (Linux/Alpine)
# Crystal: 24000000 Hz  TSC: 2808.010 MHz
set -e

echo ''
echo '+--------------------------------------------------------------+'
echo '|  Quantum Throughput Init  (AVX2/FMA3 phi-Weyl lattice)      |'
echo '+--------------------------------------------------------------+'
echo ''

if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
  echo '  [OK] AVX2 detected'
else
  echo '  [warn] No AVX2 in /proc/cpuinfo -- scalar fallback'
fi
if grep -q fma /proc/cpuinfo 2>/dev/null; then
  echo '  [OK] FMA3 detected'
fi

apk add --quiet gcc musl-dev 2>/dev/null || true

cat > /tmp/lattice_q.c << 'CEOF'
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#define PHI 1.6180339887498948482
#define N   4096
static double lat[N];
static float  lat_f[N];
static long ns_now(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec*1000000000L+t.tv_nsec;}
int main(void){
  long t0,t1; volatile double acc=0.0; int R;
  R=50000;
  t0=ns_now();
  for(int r=0;r<R;r++)
    for(int i=0;i<N;i++){double v=i*PHI;lat[i]=v-__builtin_floor(v);}
  t1=ns_now();
  double ns=(double)(t1-t0)/(R*(double)N);
  printf("  phi-Weyl fill:      %6.2f ns/slot  %6.1f Mslot/s\n",ns,(double)R*N/((t1-t0)*1e-3));
  R=10000;
  t0=ns_now();
  for(int r=0;r<R;r++)
    for(int i=0;i<N;i++){double v=PHI*(lat[i]+1.0);lat[i]=v-__builtin_floor(v);}
  t1=ns_now();
  ns=(double)(t1-t0)/(R*(double)N);
  printf("  FMA resonance step: %6.2f ns/slot  %5.1f GFLOPS\n",ns,4.0/ns);
  for(int i=0;i<N;i++)lat_f[i]=(float)lat[i];
  R=500000;
  t0=ns_now();
  for(int r=0;r<R;r++)
    for(int i=0;i<N;i++){float p=__builtin_fmodf(lat_f[i]*0.6180339887f,1.0f);
      acc+=1.0f-__builtin_fabsf(p-0.5f)*2.0f;}
  t1=ns_now();
  ns=(double)(t1-t0)/(R*(double)N);
  printf("  Analog score f32x8: %6.3f ns/slot  %5.1f Gscore/s\n",ns,1.0/ns);
  printf("  (acc=%.3f)\n",acc);
  return 0;}
CEOF

echo '  Compiling with gcc -O3 -mavx2 -mfma...'
if gcc -O3 -mavx2 -mfma -ffast-math -o /tmp/avx2_seed /tmp/lattice_q.c -lm 2>/dev/null; then
  echo '  [OK] compiled'
  /tmp/avx2_seed
  cp /tmp/avx2_seed /usr/local/bin/avx2_seed
  chmod +x /usr/local/bin/avx2_seed
  echo '  [OK] /usr/local/bin/avx2_seed installed'
else
  echo '  [warn] gcc -mavx2 failed -- check Alpine gcc version'
fi

cat > /etc/profile.d/lattice_quantum.sh << 'QEOF'
export LATTICE_BACKEND=avx2_fma
export LATTICE_SIMD_DOUBLE=4
export LATTICE_SIMD_FLOAT=8
alias quantum_seed='avx2_seed 2>/dev/null || echo avx2_seed not installed'
QEOF

echo '  [OK] /etc/profile.d/lattice_quantum.sh written'
echo '  [OK] LATTICE_BACKEND=avx2_fma active in all new shells'
echo ''
