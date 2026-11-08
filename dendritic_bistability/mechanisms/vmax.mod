: Source: Gao et al. (2021), ModelDB 249705, mod/vmax.mod.

NEURON {
   SUFFIX vmax
   RANGE vm, tpeak
}

ASSIGNED {
   v (millivolt)
   vm (millivolt)
   tpeak (ms)
}

INITIAL {
    vm = v
    tpeak = t
}

BREAKPOINT { 
   if (v>vm) { 
      vm=v
      tpeak=t 
   }
}
