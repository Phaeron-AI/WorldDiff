from __future__ import annotations

# Third Party Import(s)
from torch import Tensor

def cfg_velocity(
  v_uncond: Tensor,
  v_cond: Tensor,
  w: float
) -> Tensor:
  if v_uncond.shape != v_cond.shape: 
    raise ValueError( "v_uncond and v_cond must have the same shape" ) 
  
  if v_uncond.dtype != v_cond.dtype: 
    raise ValueError( "v_uncond and v_cond must have the same dtype" ) 
  
  if v_uncond.device != v_cond.device: 
    raise ValueError( "v_uncond and v_cond must be on the same device" )
  
  return v_uncond + w * (v_cond - v_uncond)
