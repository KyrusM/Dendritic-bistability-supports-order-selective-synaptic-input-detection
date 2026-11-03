# Model sources

## Reused code

Gao et al. (2021), *Local glutamate-mediated dendritic plateau potentials change the state of the cortical pyramidal neuron*, J. Neurophysiol. 125, 23–42. Source: [ModelDB 249705](https://modeldb.science/249705), [revision 231103b9](https://github.com/ModelDBRepository/249705/tree/231103b9b46d26cd82b3ca3dfdfbf191c19dceb3).

The following files in `dendritic_bistability/mechanisms` retain the source code from the model's `mod/` directory:

| Files | Mechanism |
|---|---|
| `na.mod`, `kv.mod` | Sodium and delayed-rectifier potassium |
| `kaprox.mod`, `kadist.mod` | A-type potassium |
| `ca.mod`, `CaT.mod` | High- and low-voltage-activated calcium |
| `Ih.mod` | HCN |
| `kBK.mod` | Calcium-activated potassium |
| `Cad.mod` | Calcium handling |

Each file identifies this distribution at the top and retains its earlier author and modification notices. `kBK.mod` also identifies its upstream source, ModelDB 168148. Shaft and soma channel densities derive from Gao's CA229 model.

Branco, Clark & Häusser (2010), *Dendritic discrimination of temporal input sequences in cortical neurons*, Science 329, 1671–1675. Source: [ModelDB 140828](https://modeldb.science/140828), [revision 1074f16c](https://github.com/ModelDBRepository/140828/tree/1074f16cfa19a1896dbd18de9d4d2223c4519379).

The files in `dendritic_bistability/branco/mechanisms`, except `sequence_release.mod`, retain the source mechanisms from `mod.files/`. The reconstruction `rc19.hoc`, parameter setup `init_biophys.hoc`, synapse locations and original readme are also retained. `init_synapses.hoc` omits only the synapse-index printout. Source headers identify these files; their original notices remain intact.

## Equations and parameters

Figure S7's `geometry.json` contains the reduced soma, one dendritic branch and 124 explicit spines derived from MICrONS cell 864691135117980637 (branch tip SWC 7140). It includes the cable mesh and the anatomical projection used in the figure. Source: MICrONS Consortium (2025), [*Functional connectomics spanning multiple areas of mouse visual cortex*](https://doi.org/10.1038/s41586-025-08790-w), Nature 640, 435–447. The reconstruction and electrical protocols are described in Supplementary Methods. Geometry-preparation code and source morphology are in `SupplementaryFigure7/Reconstruction/`.

- Boahen (2022), [*Dendrocentric learning for synthetic intelligence*](https://doi.org/10.1038/s41586-022-05340-6), Nature 612, 43–50. Source of the NMDA and GIRK voltage-gating parameters used here.
- Iacobucci & Popescu (2017). GluN1/GluN2A modal-gating responses underlie the NMDA decay waveform, temperature-scaled as in Boahen (2022).
- Figures S5 and S6 include a 40:60 head-to-shaft GIRK split, based on inhibitory synapse locations in Gemin et al. (2021), [doi:10.1371/journal.pbio.3001375](https://doi.org/10.1371/journal.pbio.3001375).
- Background inputs use locations from Gemin et al. (2021), rates from Rudolph & Destexhe (2003), AMPA conductances from Beaulieu-Laroche & Harnett (2018), and relative NMDA and GABA_A spreads from Racca et al. (2000) and Meye et al. (2013). The GABA_A distribution retains a 0.5-nS mean; the NMDA mean is 1.05 nS. Source DOIs appear in `reconstruction_background.py`.

`ampa.mod`, `nmda.mod` and `girk.mod` implement the currents described in Supplementary Methods. `front_events.mod` extends the AMPA and NMDA waveforms to repeated inputs. `branco/mechanisms/sequence_release.mod` extends the Branco transmitter protocol to repeated inputs. These implementations, the cylindrical dendrites, explicit spines, idealized somas and stimulation protocols were written for this study.

## ModelDB citation

McDougal et al. (2017), [*Twenty years of ModelDB and beyond: building essential modeling tools for the future of neuroscience*](https://doi.org/10.1007/s10827-016-0623-7), J. Comput. Neurosci. 42, 1–10.

## Source notices

Gao's source README identifies Srdjan Antic's lab and Neurosim lab as project owners, without specifying redistribution terms. The Branco source distribution does not state an explicit software license. Original author notices and references are retained.
