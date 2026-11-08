: Repeated-input extension of the Branco et al. (2010) transmitter protocol.
: https://modeldb.science/140828

TITLE Repeated glutamate release
NEURON { POINT_PROCESS SequenceRelease RANGE T, amp, dur, active }
PARAMETER { amp=2 dur=0.5 (ms) }
ASSIGNED { T active }
INITIAL { T=0 active=0 }
NET_RECEIVE(weight) {
 if (flag==0) {active=active+1 net_send(dur,1)}
 else {active=active-1}
 if (active>0) {T=amp} else {T=0}
}
