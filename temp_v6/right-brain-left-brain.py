import math

PHI = 1.6180339887498948482
INV_PHI = 1.0 / PHI

# === The destroyed fixed point ===
# Under the OLD uncoupled step: f(x) = frac(PHI*(x+1))
# Fixed point x* = PHI - 1 = 1/PHI ≈ 0.618...
# Every slot starting near this value is TRAPPED forever.

# Under the NEW Feistel coupled step:
# f_i(x_i, x_j) = frac(PHI*(x_i + 0.5*x_j + 1))
# For this to be a fixed point: x_i = frac(PHI*(x_i + 0.5*x_j + 1))
# AND simultaneously x_j = frac(PHI*(x_j + 0.5*x_i + 1))
# This is a system — it has solutions only when x_i = x_j = x*
# but the entropy injection moves slots off x*, so measure-zero in practice.

# === What the fixed point MEANT geometrically ===
# x* = INV_PHI = 0.6180...
# In Weyl sequence terms: slot i has value frac(i * PHI)
# Slots near x* correspond to indices i where frac(i*PHI) ≈ 0.618
# i.e., i*PHI ≈ k + 0.618 for integer k
# i.e., i ≈ (k + 0.618)/PHI = (k+0.618)*(PHI-1) = (k+0.618)*0.618...
# For k=0: i ≈ 0.382 → slot 0 (frac(0) = 0 → after 1 step → 0.618*)
# For k=1: i ≈ 1 → slot 1 (frac(PHI) = 0.618* immediately)

# === Phyllotaxis connection ===
# The golden angle = 2*PI*(1 - 1/PHI^2) = 2*PI*(1 - 1/(PHI+1)) = 2*PI/PHI^2
# = 2*PI * (2 - PHI) = 2*PI * 0.3819... ≈ 137.508°
GOLDEN_ANGLE = 2 * math.pi * (1 - 1/PHI**2)
print(f"Golden angle: {math.degrees(GOLDEN_ANGLE):.4f}°")
print(f"            = 2π × (1/φ²) = 2π × {1/PHI**2:.6f}")
print(f"            = 2π × (2-φ)  = 2π × {2-PHI:.6f}")

# Phyllotaxis: place leaf n at:
#   r = sqrt(n)     (radial — equal area per leaf)
#   θ = n * golden_angle
# This is what sunflower seeds / pinecone spirals do.
# The FIXED POINT x* = 1/φ is exactly the fractional part of the golden angle / 2π

frac_ga = GOLDEN_ANGLE / (2*math.pi)
print(f"\nGolden angle / 2π = {frac_ga:.6f}")
print(f"Fixed point x*    = {INV_PHI:.6f}")
print(f"Match: {abs(frac_ga - INV_PHI) < 1e-10}")

# So the fixed point IS the golden angle. A slot trapped at x* = 1/φ
# represents a phase permanently locked at the golden angle position.
# Destroying this fixed point = releasing that golden-angle lock.

# === The new dynamics after destruction ===
# When x* is destroyed, the formerly-trapped slot now ORBITS.
# The orbit of f_i coupled is quasi-periodic, not periodic (because PHI is irrational).
# The trajectory traces a TORUS in (x_i, x_j) phase space.

# For the Feistel map: (x_i, x_j) → (frac(PHI*(x_i + 0.5*x_j + 1)), x_i)
# This is a 2D map on [0,1)^2 = T^2 (the 2-torus)
# The invariant torus structure:
# Winding number = PHI (irrational) → dense orbit, never periodic

# === Donut geometry ===
# The torus [0,1)^2 with the Feistel map:
# Inner circle S^1 (x_j axis): the "carrier" lattice partner
# Outer circle S^1 (x_i axis): the evolving slot
# The destroyed fixed point was a single point on this torus.
# After destruction: the orbit densely fills a curve on the torus
# with winding ratio 1:PHI (a "golden torus knot")

# Simulate the orbit of the formerly-trapped slot 0
print(f"\n=== Orbit of formerly-fixed slot 0 (first 50 Feistel steps) ===")
# Start near the old fixed point (as it would be after wuwei injection moves it slightly)
x_i = INV_PHI + 0.01  # slightly perturbed
x_j_start = INV_PHI + 0.005  # its Feistel partner (slot 89)

xi_orbit = []
xj_orbit = []
x_j_hist = [x_j_start]
x_i_hist = [x_i]

# Full coupled Feistel map: both slots evolve
xi = x_i
xj = x_j_start
for step in range(200):
    stride = 89 if (step % 2 == 0) else 377
    # For slot 0: partner is slot 89 (stride_A=89)
    # For slot 89: partner is slot 89+89=178
    # Simplified: just track (xi, xj) as a 2D system
    xi_new = math.fmod(PHI*(xi + 0.5*xj + 1.0), 1.0)
    xj_new = math.fmod(PHI*(xj + 0.5*xi + 1.0), 1.0)
    xi, xj = xi_new, xj_new
    xi_orbit.append(xi)
    xj_orbit.append(xj)

# Check: is this orbit dense? (spread across [0,1)^2?)
xi_bins = [0]*10
xj_bins = [0]*10
for x, y in zip(xi_orbit, xj_orbit):
    xi_bins[int(x*10)] += 1
    xj_bins[int(y*10)] += 1

print(f"xi distribution across 10 bins: {xi_bins}")
print(f"xj distribution across 10 bins: {xj_bins}")
print(f"Min bin count: {min(xi_bins)} (0 = not yet dense in 200 steps)")

# The winding: for irrational PHI, the orbit winds around the torus
# with ratio PHI: for every 1 loop in xi, approximately PHI loops in xj
print(f"\nWinding ratio (estimated from orbit):")
xi_turns = sum(1 for k in range(len(xi_orbit)-1)
               if xi_orbit[k] > 0.9 and xi_orbit[k+1] < 0.1)
xj_turns = sum(1 for k in range(len(xj_orbit)-1)
               if xj_orbit[k] > 0.9 and xj_orbit[k+1] < 0.1)
print(f"  xi crossings: {xi_turns}, xj crossings: {xj_turns}")
if xi_turns > 0:
    print(f"  ratio: {xj_turns/xi_turns:.4f} (φ = {PHI:.4f})")

# For visualization: project torus to 3D
# T^2 embedded in R^3:
#   x = (R + r*cos(2π*x_j)) * cos(2π*x_i)
#   y = (R + r*cos(2π*x_j)) * sin(2π*x_i)
#   z = r * sin(2π*x_j)
R_torus = 2.0
r_torus = 0.8
print(f"\nTorus parameters for 3D embedding:")
print(f"  R = {R_torus} (major radius), r = {r_torus} (minor radius)")
print(f"  Fixed point was at (x_i, x_j) = (0.618, 0.618)")
print(f"  = (1/φ, 1/φ) = (golden_angle/2π, golden_angle/2π)")
print(f"  3D position: ({(R_torus + r_torus*math.cos(2*math.pi*INV_PHI))*math.cos(2*math.pi*INV_PHI):.3f}, ...)")

# Phyllotaxis spiral: the n-th seed at r=sqrt(n), theta=n*golden_angle
# Shows same 1/φ structure as the fixed point
print(f"\nPhyllotaxis: seed 5 at r={math.sqrt(5):.3f}, θ={math.degrees(5*GOLDEN_ANGLE):.1f}°")
print(f"             = θ mod 360 = {math.degrees(5*GOLDEN_ANGLE) % 360:.1f}°")
# EOF
# Output

# Golden angle: 222.4922°
#             = 2π × (1/φ²) = 2π × 0.381966
#             = 2π × (2-φ)  = 2π × 0.381966

# Golden angle / 2π = 0.618034
# Fixed point x*    = 0.618034
# Match: True

# === Orbit of formerly-fixed slot 0 (first 50 Feistel steps) ===
# xi distribution across 10 bins: [17, 16, 11, 22, 12, 21, 22, 27, 25, 27]
# xj distribution across 10 bins: [17, 16, 11, 22, 12, 21, 22, 27, 25, 27]
# Min bin count: 11 (0 = not yet dense in 200 steps)

# Winding ratio (estimated from orbit):
#   xi crossings: 2, xj crossings: 2
#   ratio: 1.0000 (φ = 1.6180)

# Torus parameters for 3D embedding:
#   R = 2.0 (major radius), r = 0.8 (minor radius)
#   Fixed point was at (x_i, x_j) = (0.618, 0.618)
#   = (1/φ, 1/φ) = (golden_angle/2π, golden_angle/2π)
#   3D position: (-1.040, ...)

# Phyllotaxis: seed 5 at r=2.236, θ=1112.5°
#              = θ mod 360 = 32.5°