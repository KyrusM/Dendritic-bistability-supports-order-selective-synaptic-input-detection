: GIRK voltage gating: Boahen (2022).
: https://doi.org/10.1038/s41586-022-05340-6

TITLE GIRK current

COMMENT
Apply inward rectification to a maintained GIRK conductance.
g_at_e specifies conductance at the reversal potential.
ENDCOMMENT

NEURON {
    SUFFIX girk
    NONSPECIFIC_CURRENT i
    RANGE g_at_e, e, vhalf, vslope, gate, g, i
}

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S) = (siemens)
}

PARAMETER {
    g_at_e = 0 (S/cm2)
    e = -90 (mV)
    vhalf = -80 (mV)
    vslope = -10 (mV)
}

ASSIGNED {
    v (mV)
    gate (1)
    g (S/cm2)
    i (mA/cm2)
}

BREAKPOINT {
    gate = (1 + exp(-(e - vhalf)/vslope)) /
           (1 + exp(-(v - vhalf)/vslope))
    g = g_at_e * gate
    i = g * (v - e)
}
