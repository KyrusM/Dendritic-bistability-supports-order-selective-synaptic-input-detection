"""Mean-preserving lognormal variation and input orders for Figure S6."""

import hashlib

import numpy as np


FAMILIES = ("NMDA", "AMPA", "GIRK", "SPACING", "NECK", "JOINT")
SDS = (0.0, 0.02, 0.05, 0.10, 0.20)
# GIRK at −90 mV: nS per head and nS per micrometer of shaft.
GIRK_CONDUCTANCES = {
    "head": (4.5, 0.0),
    "split": (24 / 13, 36 / 110),
    "shaft": (0.0, 0.45),
}
PLACEMENTS = tuple(GIRK_CONDUCTANCES)
SAMPLES = 32
SHUFFLES = 6


def structured_orders():
    """Return ordered inputs, all eight adjacent swaps and reversed inputs."""
    ordered = list(range(1, 10))
    cases = [dict(name="ordered", kind="ordered", order=ordered)]
    for i in range(8):
        swapped = ordered.copy()
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        cases.append(dict(name=f"swap_{i+1}_{i+2}", kind="swap", order=swapped))
    return cases + [dict(name="reversed", kind="reverse", order=ordered[::-1])]


def random_seed(identity):
    """Convert a condition identity to a reproducible random seed."""
    return int(hashlib.sha256(identity.encode()).hexdigest()[:32], 16)


def parameter_factors(family, sd, draw):
    """Sample independent parameter multipliers, holding the four seed spines fixed."""
    factors = {name: np.ones(13) for name in ("NMDA", "AMPA", "GIRK", "NECK")}
    factors["SPACING"] = np.ones(8)
    if not sd:
        return {name: values.tolist() for name, values in factors.items()}
    variance = np.log1p(sd**2)
    for name, values in factors.items():
        if family not in (name, "JOINT"):
            continue
        identity = f"S6-lognormal-20261101|{name}|sd={sd:g}|draw={draw}"
        rng = np.random.default_rng(random_seed(identity))
        size = 8 if name == "SPACING" else 9
        sampled = rng.lognormal(-variance / 2, np.sqrt(variance), size)
        values[:] = sampled if name == "SPACING" else np.r_[np.ones(4), sampled]
    return {name: values.tolist() for name, values in factors.items()}


def shuffled_orders(identity):
    """Draw six shuffled sequences with replacement, excluding the structured orders."""
    excluded = {tuple(case["order"]) for case in structured_orders()}
    cases = []
    for presentation in range(SHUFFLES):
        seed = random_seed(f"{identity}|shuffle={presentation}")
        rng = np.random.default_rng(seed)
        while True:
            order = rng.permutation(np.arange(1, 10)).tolist()
            if tuple(order) not in excluded:
                break
        cases.append(dict(name=f"shuffle_{presentation+1:02}", kind="shuffle", order=order,
                          random_seed=str(seed)))
    return cases


def variability_samples():
    """Build nominal controls and 32 dendrites per nonzero SD, parameter group and placement."""
    samples = []
    conditions = [("BASELINE", 0.0, 0)]
    conditions += [(family, sd, draw) for draw in range(SAMPLES)
                   for sd in SDS[1:] for family in FAMILIES]
    for family, sd, draw in conditions:
        for placement in PLACEMENTS:
            # Match input orders between head-only and split GIRK.
            matched = "head" if placement == "split" else placement
            identity = f"S6-lognormal-20261101|{matched}|{family}|sd={sd:g}|draw={draw}"
            samples.append(dict(
                name=f"{placement}_{family}_sd{100 * sd:g}_{draw:02}",
                placement=placement, family=family, sd=sd, realization=draw,
                factors=parameter_factors(family, sd, draw),
                random_stream_identity=identity,
                orders=structured_orders() + shuffled_orders(identity),
            ))
    return samples
