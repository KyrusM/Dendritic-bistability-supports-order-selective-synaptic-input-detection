: NMDA voltage gating: Boahen (2022), https://doi.org/10.1038/s41586-022-05340-6.
: Decay constants: Iacobucci & Popescu (2017), temperature-scaled as in Boahen (2022).

TITLE NMDA current

COMMENT
Apply the input-triggered NMDA temporal gate and voltage-dependent magnesium block.
g_at_e specifies conductance at reversal when the temporal gate equals one.
ENDCOMMENT

NEURON {
    SUFFIX nmda
    NONSPECIFIC_CURRENT i
    RANGE g_at_e, e, vhalf, vslope, onset
    RANGE tau_rise, tau_decay_fast, tau_decay_mid, tau_decay_slow
    RANGE fast_weight, normalization, time_gate, voltage_gate, g, i
}

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S) = (siemens)
}

PARAMETER {
    g_at_e = 0 (S/cm2)
    e = 0 (mV)
    vhalf = -23.7 (mV)
    vslope = 12.5 (mV)
    onset = 1e9 (ms)

    tau_rise = 1.4 (ms)
    tau_decay_fast = 12.576284467576945 (ms)
    tau_decay_mid = 57.262231405563114 (ms)
    tau_decay_slow = 371.9369236155735 (ms)
    fast_weight = 0.5 (1)
    normalization = 2.1690809854229562 (1)
}

ASSIGNED {
    v (mV)
    age (ms)
    time_gate (1)
    voltage_gate (1)
    g (S/cm2)
    i (mA/cm2)
}

BREAKPOINT {
    age = t - onset
    if (age < 0) {
        time_gate = 0
    } else {
        time_gate = (
            fast_weight * exp(-age/tau_decay_fast)
            + exp(-age/tau_decay_mid)
            + exp(-age/tau_decay_slow)
            - (fast_weight + 2) * exp(-age/tau_rise)
        ) / normalization
        if (time_gate < 0) {
            time_gate = 0
        }
    }

    voltage_gate = (1 + exp(-(e - vhalf)/vslope)) /
                   (1 + exp(-(v - vhalf)/vslope))
    g = g_at_e * time_gate * voltage_gate
    i = g * (v - e)
}
