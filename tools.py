"""Simulated healthcare-agent tools."""

from patient import Patient


def schedule_appointment(patient: Patient) -> dict[str, str]:
    """Return a simulated consultation booking for the supplied patient."""
    if not isinstance(patient, Patient):
        raise TypeError("schedule_appointment requires a Patient instance")

    return {
        "status": "confirmed",
        "doctor": "Dr. Sarah Smith",
        "date": "Tomorrow",
        "time": "11:00 AM",
    }