#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vasp_analyzer.py — Automatic PDOS-peak orbital identification and WAVECAR export.

Workflow
--------
1. Read PDOS data and locate the dominant peak (e.g. d_z² of Fe).
2. Determine the majority-spin channel from OSZICAR.
3. Parse OUTCAR band structure to find the band closest to the peak energy.
4. Export the real-space probability density |ψ|² from WAVECAR via *vaspwfc*.

The exported ``_r.vasp`` file (CHGCAR format) can be fed directly into
``compute_z_centroid.py`` or ``pca_orbital_analysis.py``.

Dependencies
------------
- numpy, pandas
- vaspwfc (https://github.com/QijingZheng/VaspBandUnfolding)
"""

import re
import numpy as np
import pandas as pd
from vaspwfc import vaspwfc


class VaspPDOSAnalyzer:
    """
    Analyse VASP outputs to locate the PDOS-peak band and export its
    real-space probability density.
    """

    def __init__(
        self,
        outcar="OUTCAR",
        doscar="DOSCAR",
        oszicar="OSZICAR",
        pdos_file="PDOS_USER.dat",
        wavecar="WAVECAR",
        poscar="POSCAR",
    ):
        self.outcar_path = outcar
        self.doscar_path = doscar
        self.oszicar_path = oszicar
        self.pdos_file_path = pdos_file
        self.wavecar_path = wavecar
        self.poscar_path = poscar

        self.fermi_energy = None
        self.mag_state = None
        self.bands_df = None
        self.max_pdos_info = {}
        self.target_band_info = {}

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------
    def _parse_outcar_bands(self):
        """Parse OUTCAR for band eigenvalues and occupations."""
        spin_regex = re.compile(r"spin component\s+([12])")
        kpoint_regex = re.compile(r"k-point\s+([0-9]+)\s*:.*")
        band_regex = re.compile(r"^\s*([0-9]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")

        parsed_data = {}
        current_spin = None
        current_kpoint = None

        with open(self.outcar_path, "r") as f:
            for line in f:
                spin_match = spin_regex.search(line)
                if spin_match:
                    current_spin = int(spin_match.group(1))
                    if current_spin not in parsed_data:
                        parsed_data[current_spin] = {}
                    continue
                if current_spin:
                    kpoint_match = kpoint_regex.search(line)
                    if kpoint_match:
                        current_kpoint = int(kpoint_match.group(1))
                        if current_kpoint not in parsed_data[current_spin]:
                            parsed_data[current_spin][current_kpoint] = []
                        continue
                if current_kpoint:
                    band_match = band_regex.match(line)
                    if band_match:
                        band_index, energy, occupation = map(
                            float, band_match.groups()
                        )
                        parsed_data[current_spin][current_kpoint].append(
                            (int(band_index), energy, occupation)
                        )

        records = []
        for spin, kpoints in parsed_data.items():
            for kpoint_idx, bands in kpoints.items():
                for band_data in bands:
                    records.append(
                        {
                            "spin": spin,
                            "kpoint_index": kpoint_idx,
                            "band_index": band_data[0],
                            "energy": band_data[1],
                            "occupation": band_data[2],
                        }
                    )
        self.bands_df = pd.DataFrame(records)
        print(f"Parsed OUTCAR: {len(self.bands_df)} band records found.")

    def _read_dos_and_fermi(self):
        """Read Fermi energy from DOSCAR and PDOS data from *pdos_file*."""
        with open(self.doscar_path, "r") as f:
            self.fermi_energy = float(f.readlines()[5].split()[3])
        self.pdos_data = np.loadtxt(self.pdos_file_path, comments="#")
        print(f"Fermi energy: {self.fermi_energy:.4f} eV")

    def _determine_spin_state(self):
        """Determine majority-spin channel from OSZICAR magnetic moment."""
        with open(self.oszicar_path, "r") as f:
            last_line = f.readlines()[-1]
            mag_moment = float(last_line.split()[9])
        self.mag_state = 2 if mag_moment > 0.0 else 1
        print(
            f"Magnetic moment: {mag_moment:.4f}, "
            f"majority-spin channel: {self.mag_state}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def find_target_band_from_pdos(self):
        """
        Locate the PDOS peak and match it to the nearest OUTCAR band.

        Returns
        -------
        success : bool
        """
        self._read_dos_and_fermi()
        self._determine_spin_state()
        self._parse_outcar_bands()

        energy_rel_fermi = self.pdos_data[:, 0]
        pdos_up = self.pdos_data[:, 1]
        pdos_down = self.pdos_data[:, 2]

        pdos_channel = pdos_up if self.mag_state == 1 else pdos_down

        max_index = np.argmax(np.abs(pdos_channel))
        peak_energy_rel_fermi = energy_rel_fermi[max_index]
        self.max_pdos_info = {
            "peak_energy": peak_energy_rel_fermi + self.fermi_energy,
            "peak_pdos_value": pdos_channel[max_index],
            "spin_channel": self.mag_state,
        }
        print(
            f"PDOS peak at absolute energy: "
            f"{self.max_pdos_info['peak_energy']:.4f} eV"
        )

        target_energy = self.max_pdos_info["peak_energy"]
        df_spin_filtered = self.bands_df[
            self.bands_df["spin"] == self.mag_state
        ].copy()

        if df_spin_filtered.empty:
            print(
                f"Error: no band data for spin channel {self.mag_state}."
            )
            return False

        df_spin_filtered["abs_diff"] = (
            df_spin_filtered["energy"] - target_energy
        ).abs()
        min_diff_idx = df_spin_filtered["abs_diff"].idxmin()
        closest_band_series = df_spin_filtered.loc[min_diff_idx]

        self.target_band_info = closest_band_series.to_dict()
        print("Matched band:")
        print(pd.DataFrame([self.target_band_info]).to_string(index=False))
        return True

    def export_wavefunction(self, output_prefix=None):
        """
        Export |ψ|² for the target band from WAVECAR to a ``_r.vasp`` file.

        Parameters
        ----------
        output_prefix : str, optional
            File-name prefix. Defaults to
            ``prob_density_s{spin}_b{band}``.
        """
        if not self.target_band_info:
            print("Error: run find_target_band_from_pdos() first.")
            return

        band_idx = int(self.target_band_info["band_index"])
        spin_idx = int(self.target_band_info["spin"])

        if output_prefix is None:
            output_prefix = f"prob_density_s{spin_idx}_b{band_idx}"

        print(
            f"\nExporting from WAVECAR (spin={spin_idx}, band={band_idx})..."
        )
        pswfc = vaspwfc(self.wavecar_path, lgamma=True)
        phi_complex = pswfc.wfc_r(
            ispin=spin_idx, ikpt=1, iband=band_idx, ngrid=pswfc._ngrid * 2
        )
        probability_density = np.abs(phi_complex) ** 2
        pswfc.save2vesta(
            probability_density,
            poscar=self.poscar_path,
            prefix=output_prefix,
            lreal=True,
        )
        print(f"Saved to {output_prefix}_r.vasp")


# ======================================================================
if __name__ == "__main__":
    analyzer = VaspPDOSAnalyzer()
    success = analyzer.find_target_band_from_pdos()
    if success:
        analyzer.export_wavefunction()
