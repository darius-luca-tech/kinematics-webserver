from __future__ import annotations
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
import utils.fk as fk

app = FastAPI(
    title="Kinematics",
    description="Compute end-effector pose and per-link positions from 6 joint angles.",
    version="1.0.0"
)

class FKRequest(BaseModel):
    
    L1: Optional[float] = None
    L2: Optional[float] = None
    L3: Optional[float] = None
    L4: Optional[float] = None
    L5: Optional[float] = None
    L6: Optional[float] = None

    angles: Optional[list[float]] = Field(
        default=None,
        description="Alternative to L1...L6: [L1, L2, L3, L4, L5, L6]"
    )
    convention: Optional[str] = Field(
        default=None,
        description="Tool-frame convention: 'urdf' or 'diagram'."
    )

    @model_validator(mode="after")
    def _resolve(self):
        if self.angles is not None:
            if len(self.angles) != 6:
                raise ValueError("'angles' must contain exactly 6 values [L1..L6]")
            self.L1, self.L2, self.L3, self.L4, self.L5, self.L6 = self.angles
        missing = [j for j in fk.JOINTS if getattr(self, j) is None]
        if missing:
            raise ValueError(
                f"Missing angle(s): {missing}. Provide all of L1..L6, "
                f"or an 'angles' list of 6 values."
            )
        if self.convention is not None and self.convention not in fk.CONVENTIONS:
            raise ValueError(
                f"convention must be one of {fk.CONVENTIONS}, got {self.convention!r}"
            )
        return self
 
    def as_marlin_dict(self) -> dict:
        return {j: float(getattr(self, j)) for j in fk.JOINTS}
 
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"L1": 0, "L2": -90, "L3": 180, "L4": 0, "L5": 0, "L6": 180},
                {"angles": [45, -40, 180, 0, 0, 180]},
            ]
        }
    }

def _range_warnings(marlin: dict) -> list[str]:
    warns = []

    for j in fk.JOINTS:
        lim = fk.JOINT_CONFIG[j]["marlin"]
        v = marlin[j]

        if v < lim["min"] or v > lim["max"]:
            warns.append(
                f"{j}={v} is outside its range [{lim['min']}, {lim['max']}]"
            )
    
    return warns

def _compute(marlin: dict, convention: str = None) -> dict:
    pose = fk.fk_pose(marlin, convention)
    frames = fk.fk_all_frames(marlin, convention)

    link_names = ["base", "L1", "L2", "L3", "L4", "L5", "L6"]
    links = [
        {"name": name, "position_mm": (T[:3, 3] * 1000).tolist()}
        for name, T in zip(link_names, frames)
    ]

    yaw, pitch, roll = pose["euler_rel_deg"]

    return {
        "input_marlin_deg": marlin,
        "convention": pose["convention"],
        "end_effector": {
            "position_mm": pose["position_mm"],
            "orientation": {
                "euler_rel_deg": {"yaw": yaw, "pitch": pitch, "roll": roll},
                "quaternion_xyzw": pose["quaternion"],
                "tool_axis_world": pose["tool_axis"],
                "R_abs": pose["R_abs"].tolist(),
                "R_rel": pose["R_rel"].tolist(),
            },
        },
        "links": links,
        "warnings": _range_warnings(marlin),
    }

@app.post("/fk")
def fk_post(req: FKRequest):
    try:
        return _compute(req.as_marlin_dict(), req.convention)
    except Exception as exc:  # numerical / config errors
        raise HTTPException(status_code=500, detail=str(exc))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
