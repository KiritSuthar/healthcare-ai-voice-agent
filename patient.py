"""Patient data supplied to an individual voice-agent session."""

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Patient:
    """The patient facts permitted in the conversation context."""

    name: str
    phone: str
    blood_glucose: float
    hba1c: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Patient name must not be empty")
        if not isinstance(self.phone, str) or not self.phone.strip():
            raise ValueError("Patient phone must not be empty")
        if not math.isfinite(self.blood_glucose) or self.blood_glucose < 0:
            raise ValueError("Blood glucose must be a finite, non-negative number")
        if not math.isfinite(self.hba1c) or not 0 <= self.hba1c <= 100:
            raise ValueError("HbA1c must be a finite percentage from 0 to 100")

    @classmethod
    def from_metadata(cls, metadata: str) -> "Patient":
        """Parse and validate the patient JSON passed with the LiveKit job."""
        if not metadata.strip():
            raise ValueError("LiveKit job metadata must contain patient JSON")

        try:
            data: Any = json.loads(metadata)
        except json.JSONDecodeError as error:
            raise ValueError("LiveKit job metadata must be valid JSON") from error

        if not isinstance(data, dict):
            raise ValueError("Patient metadata must be a JSON object")

        required_fields = ("name", "phone", "blood_glucose", "hba1c")
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValueError("Missing patient fields: " + ", ".join(missing))

        try:
            if not isinstance(data["name"], str) or not isinstance(data["phone"], str):
                raise TypeError("name and phone must be strings")
            if isinstance(data["blood_glucose"], bool) or isinstance(data["hba1c"], bool):
                raise TypeError("biomarker values must be numbers")
            name = data["name"].strip()
            phone = data["phone"].strip()
            blood_glucose = float(data["blood_glucose"])
            hba1c = float(data["hba1c"])
        except (TypeError, ValueError) as error:
            raise ValueError("Patient fields have invalid types or values") from error

        return cls(
            name=name,
            phone=phone,
            blood_glucose=blood_glucose,
            hba1c=hba1c,
        )