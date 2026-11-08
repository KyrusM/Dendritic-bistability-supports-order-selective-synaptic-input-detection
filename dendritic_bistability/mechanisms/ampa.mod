TITLE AMPA current

COMMENT
Apply an input-triggered AMPA alpha function with peak conductance g_peak.
ENDCOMMENT

NEURON {
    SUFFIX ampa
    NONSPECIFIC_CURRENT i
    RANGE g_peak, e, tau, onset, alpha, g, i
}

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S) = (siemens)
}

PARAMETER {
    g_peak = 0 (S/cm2)
    e = 0 (mV)
    tau = 0.5 (ms)
    onset = 1e9 (ms)
}

ASSIGNED {
    v (mV)
    age (ms)
    alpha (1)
    g (S/cm2)
    i (mA/cm2)
}

BREAKPOINT {
    age = t - onset
    if (age >= 0) {
        alpha = (age/tau) * exp(1 - age/tau)
    } else {
        alpha = 0
    }
    g = g_peak * alpha
    i = g * (v - e)
}
