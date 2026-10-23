"""Parametric degradation of camera, LiDAR and radar.

Modules
-------
severity  the five-level severity ladder, interpolated from the config
camera    Koschmieder scattering, rain streaks, exposure loss
lidar     attenuation, spurious back-scatter returns, beam dilation
radar     clutter, angular smearing, speckle

The degradation is applied *after* acquisition rather than by re-simulating
weather, so that the severity axis is explicit, reproducible and cheap. The
principled physical alternative for LiDAR is the fog simulation of Hahner et
al. (ICCV 2021), which is what the paper cites; the parametric form here is
the controlled stand-in used to sweep severity on real recordings.
"""

from __future__ import annotations

__all__ = ["severity", "camera", "lidar", "radar"]
