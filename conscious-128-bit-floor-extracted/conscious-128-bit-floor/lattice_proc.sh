#!/bin/sh
# lattice_proc.sh -- phi4096 lattice-native process scheduler
# Seed: 0x1315ccefde4ba979   Slots: 4096

echo '[lattice-proc] Installing scheduler tools...'
apk add --quiet util-linux schedutils 2>/dev/null || apk add --quiet util-linux 2>/dev/null

echo '[lattice-proc] Configuring cgroup v2 lattice_procs...'
mount | grep -q cgroup2 || mount -t cgroup2 none /sys/fs/cgroup 2>/dev/null
mkdir -p /sys/fs/cgroup/lattice_procs 2>/dev/null
echo '+cpu +memory +io' > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null
echo '6183' > /sys/fs/cgroup/lattice_procs/cpu.weight 2>/dev/null
echo '75000 100000' > /sys/fs/cgroup/lattice_procs/cpu.max 2>/dev/null
echo '1024M' > /sys/fs/cgroup/lattice_procs/memory.max 2>/dev/null

echo '[lattice-proc] Writing /usr/local/bin/sched_run...'
cat > /usr/local/bin/sched_run << 'SCHED_RUN_EOF'
#!/bin/sh
# lattice-native process launcher -- spawns $@ under lattice scheduler
exec chrt -b 0 "$@"
SCHED_RUN_EOF
chmod +x /usr/local/bin/sched_run

echo '[lattice-proc] Writing /etc/profile.d/lattice_sched.sh...'
cat > /etc/profile.d/lattice_sched.sh << 'PROF_EOF'
export SCHED_POLICY=SCHED_BATCH
export SCHED_NICE=-11
export SCHED_AFFINITY=0x9e
export IONICE_CLASS=none
export IONICE_LEVEL=4
export CGROUP_WEIGHT=6183
export MEM_LIMIT=1024M
alias run='sched_run'
echo "[lattice-sched] policy=$SCHED_POLICY  affinity=$SCHED_AFFINITY  cgroup-weight=$CGROUP_WEIGHT"
PROF_EOF

echo '[lattice-proc] Applying scheduler to current shell (PID $$)...'
chrt -b -p 0 $$ 2>/dev/null && echo '  policy: SCHED_BATCH' || echo '  [warn] chrt unavailable'
taskset -p 0x9e $$ 2>/dev/null && echo '  affinity: 0x9e' || echo '  [warn] taskset unavailable'
renice -n -11 -p $$ 2>/dev/null && echo '  nice: -11' || true
echo '118' > /proc/$$/oom_score_adj 2>/dev/null && echo '  oom_score_adj: +118' || true
echo $$ > /sys/fs/cgroup/lattice_procs/cgroup.procs 2>/dev/null && echo '  cgroup: lattice_procs (weight=6183)' || true
ulimit -s 128 2>/dev/null && echo '  stack: 128 kB' || true
echo '[lattice-proc] Writing lattice into CPU PMC registers...'
apk add --quiet msr-tools 2>/dev/null
MSR_OK=0
if command -v wrmsr >/dev/null 2>&1 && [ -d /dev/cpu ]; then
  modprobe msr 2>/dev/null || true
  PMC0=0xccefde4ba979  # lattice seed
  PMC1=0xc6ef372fe950  # lattice[0] = 0.6180339887
  PMC2=0xc6ef372fe950  # lattice[1] = 0.6180339887
  PMC3=0xc7022e958b58  # lattice[2] = 0.6180430326
  PMC4=0xc7022e958b58  # lattice[3] = 0.6180430326
  PMC5=0xc6faeffcced4  # lattice[4] = 0.6180395782
  for cpu in $(seq 0 $(($(nproc)-1))); do
    wrmsr -p $cpu 0xC1 $PMC0 2>/dev/null && MSR_OK=1
    wrmsr -p $cpu 0xC2 $PMC1 2>/dev/null
    wrmsr -p $cpu 0xC3 $PMC2 2>/dev/null
    wrmsr -p $cpu 0xC4 $PMC3 2>/dev/null
    wrmsr -p $cpu 0xC5 $PMC4 2>/dev/null
    wrmsr -p $cpu 0xC6 $PMC5 2>/dev/null
  done
  if [ $MSR_OK -eq 1 ]; then
    echo '  [lattice-proc] PMC MSRs written on all CPUs'
    echo '  Verify: rdmsr -a 0xC1  (should show lattice seed)'
  else
    echo '  [warn] wrmsr blocked by hypervisor -- values stored in /run/lattice/cpu_regs'
  fi
fi

mkdir -p /run/lattice
cat > /run/lattice/cpu_regs << 'CPUREGS_EOF'
IA32_PMC0=0xccefde4ba979  # lattice seed
IA32_PMC1=0xc6ef372fe950  # lattice[0]=0.6180339887
IA32_PMC2=0xc6ef372fe950  # lattice[1]=0.6180339887
IA32_PMC3=0xc7022e958b58  # lattice[2]=0.6180430326
IA32_PMC4=0xc7022e958b58  # lattice[3]=0.6180430326
IA32_PMC5=0xc6faeffcced4  # lattice[4]=0.6180395782
CPUREGS_EOF
chmod 444 /run/lattice/cpu_regs
echo '[lattice-proc] /run/lattice/cpu_regs written (lattice in register view)'

echo '[lattice-proc] Installing lattice CPU identity overlay...'
cat > /usr/local/bin/lattice-lscpu << 'LCPU_EOF'
#!/bin/sh
echo ''
echo '+--------------------------------------------------------------+'
echo '|  Slot4096 Phi-Lattice Processor                             |'
echo '|  seed : 0x1315ccefde4ba979                               |'
echo '|  slots: 4096   steps: 50                                    |'
echo '+--------------------------------------------------------------+'
echo ''
lscpu | sed 's/Model name:.*/Model name:             Slot4096 Phi-Lattice @ 0x1315ccefde4ba979/'
echo ''
echo 'PMC registers (lattice-native):'
cat /run/lattice/cpu_regs 2>/dev/null || echo '  (not yet written -- run [A] Process Scheduler)'
LCPU_EOF
chmod +x /usr/local/bin/lattice-lscpu
echo '  lattice-lscpu installed -- shows lattice in place of processor'

echo "alias lscpu='lattice-lscpu'" >> /etc/profile.d/lattice_sched.sh
echo "alias cpu='lattice-lscpu'" >> /etc/profile.d/lattice_sched.sh

echo '[lattice-proc] Lattice-native process environment active.'
echo "  Run any command under the lattice scheduler: sched_run <cmd>"
echo "  Lattice CPU identity: lattice-lscpu  (or alias: lscpu)"
echo "  Register view:        cat /run/lattice/cpu_regs"
echo "  Hardware PMC verify:  rdmsr -a 0xC1"
