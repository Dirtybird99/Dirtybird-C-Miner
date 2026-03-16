# DirtyBird Miner Performance Tuning Guide

## Current Status (2026-01-28)

### Measured Performance
- **Single-thread hashrate**: 1573 H/s
- **Theoretical 24-thread max**: ~37.7 KH/s
- **Actual 24-thread (60s)**: ~19.5 KH/s (thermal throttling)
- **Target**: 22 KH/s stable

### Optimization Flags
| Flag | Status | Impact |
|------|--------|--------|
| USE_DEROBWT | DISABLED | Was causing 14.6% regression |
| USE_SELECTIVE_MEMCPY | DISABLED | Has hash correctness bugs |
| USE_CRYPTOGAMS_RC4_DUAL | ENABLED | Dual-state for SPSA compatibility |

## Achieving 22 KH/s Stable

### The Math
- 22,000 H/s ÷ 1573 H/s/thread = **14 threads at 100% efficiency**
- With 25% thermal loss: need ~18 threads
- With 40% thermal loss: need ~24 threads

### Recommended Test Configurations

Test each for 120+ seconds to measure sustained (not peak) hashrate:

```bash
# Config 1: 16 threads, P-core priority
./dirtybird-miner-cpu.exe --dero --threads 16 --differential-affinity 1 \
  --daemon-address YOUR_POOL --wallet YOUR_WALLET --mine-time 120

# Config 2: 18 threads, P-core priority
./dirtybird-miner-cpu.exe --dero --threads 18 --differential-affinity 1 \
  --daemon-address YOUR_POOL --wallet YOUR_WALLET --mine-time 120

# Config 3: 20 threads, balanced affinity
./dirtybird-miner-cpu.exe --dero --threads 20 --differential-affinity 3 \
  --daemon-address YOUR_POOL --wallet YOUR_WALLET --mine-time 120

# Config 4: 24 threads, default (baseline)
./dirtybird-miner-cpu.exe --dero --threads 24 --no-lock \
  --daemon-address YOUR_POOL --wallet YOUR_WALLET --mine-time 120
```

### Affinity Modes
- `0`: Default OS scheduling
- `1`: P-cores first (recommended for hybrid CPUs)
- `2`: Physical cores only
- `3`: Balanced across core types

## Hardware Considerations

The 22 KH/s target may require hardware improvements:

1. **CPU Cooling**: Better thermal solution reduces throttling
2. **Power Limits**: Ensure PL1/PL2 are not limiting
3. **Ambient Temperature**: Cooler room = better sustained performance

## Known Issues

### Selective Memcpy Bug
The USE_SELECTIVE_MEMCPY optimization fails hash tests for inputs with lengths
where `length % 4 == 0 or 3`. Investigation pending.

### Pool Connection Issues
Some pools may reject WebSocket connections. Try multiple pools:
- example-pool.invalid:10300
- Your local node (if running)

## Comparison with DeroLuna

| Metric | DirtyBird | DeroLuna |
|--------|-----------|----------|
| Peak hashrate | 26+ KH/s | 25+ KH/s |
| 60s sustained | 19.5 KH/s | 23.3 KH/s |
| Thermal drop | 7 KH/s in 45s | 2-3 KH/s in 45s |
| Binary size | 10.8 MB | 3.3 MB |
| Dev fee | 0% | 10% |

DeroLuna maintains better sustained hashrate due to smaller binary (better
instruction cache) and different thermal profile.
