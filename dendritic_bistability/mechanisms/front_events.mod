: NMDA voltage gating: Boahen (2022), https://doi.org/10.1038/s41586-022-05340-6.
: Decay constants: Iacobucci & Popescu (2017), temperature-scaled as in Boahen (2022).

TITLE Repeated glutamate inputs

COMMENT
Sum repeated AMPA and NMDA inputs at one spine.
AMPA inputs add; the NMDA temporal gate is capped at one when cap_nmda is enabled.
Conductances are in nS at AMPA peak or NMDA reversal and peak temporal activation.
ENDCOMMENT

NEURON {
    POINT_PROCESS FrontEvents
    NONSPECIFIC_CURRENT i
    RANGE nmda_ns, ampa_ns, e, cap_nmda, ta
    RANGE nmda_gate, ampa_gate, nmda_g_ns, ampa_g_ns, i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
}

PARAMETER {
    nmda_ns = 2.4
    ampa_ns = 0.4
    e = 0 (mV)
    cap_nmda = 1
    tr = 1.4 (ms)
    tf = 12.576284467576945 (ms)
    tm = 57.262231405563114 (ms)
    ts = 371.9369236155735 (ms)
    norm = 2.1690809854229562
    ta = 0.5 (ms)
}

STATE { f m s r ax ay }
ASSIGNED { v (mV) i (nA) nmda_gate ampa_gate nmda_g_ns ampa_g_ns vg }

INITIAL {
    f = 0
    m = 0
    s = 0
    r = 0
    ax = 0
    ay = 0
}

BREAKPOINT {
    SOLVE states METHOD derivimplicit
    nmda_gate = (f + m + s - r) / norm
    if (nmda_gate < 0) { nmda_gate = 0 }
    if (cap_nmda > 0.5 && nmda_gate > 1) { nmda_gate = 1 }
    ampa_gate = exp(1) * ay
    vg = (1 + exp(-23.7 / 12.5)) / (1 + exp(-(v + 23.7) / 12.5))
    nmda_g_ns = nmda_ns * nmda_gate * vg
    ampa_g_ns = ampa_ns * ampa_gate
    i = (nmda_g_ns + ampa_g_ns) * (v - e) * 0.001
}

COMMENT
Evolve the NMDA decay components and AMPA alpha function between inputs.
ENDCOMMENT
DERIVATIVE states {
    f' = -f / tf
    m' = -m / tm
    s' = -s / ts
    r' = -r / tr
    ax' = -ax / ta
    ay' = (ax - ay) / ta
}

COMMENT
Add the temporal-gate components for each glutamate input.
ENDCOMMENT
NET_RECEIVE(weight) {
    f = f + 0.5 * weight
    m = m + weight
    s = s + weight
    r = r + 2.5 * weight
    ax = ax + weight
}
