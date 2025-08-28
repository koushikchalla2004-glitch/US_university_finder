import os, requests
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SCORECARD_API_KEY")
BASE = "https://api.data.gov/ed/collegescorecard/v1/schools"

# Fields we need. Using "latest" keeps it current automatically.
FIELDS = [
    "id",
    "school.name",
    "school.city",
    "school.state",
    "school.school_url",
    "location.lat",
    "location.lon",
    # Selectivity
    "latest.admissions.admission_rate.overall",
    # Costs (per year)
    "latest.cost.tuition.in_state",
    "latest.cost.tuition.out_of_state",
    "latest.cost.roomboard.oncampus",
    "latest.cost.other_on_campus",
    "latest.cost.booksupply",
    # Programs (4-digit CIP) – returns arrays when present
    "latest.programs.cip_4_digit.code",
    "latest.programs.cip_4_digit.title",
    "latest.programs.cip_4_digit.credential.level",
]

DEFAULT_PER_PAGE = 100

class ScorecardClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise RuntimeError("Missing SCORECARD_API_KEY in environment")

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "api_key": self.api_key,
            "per_page": params.pop("per_page", DEFAULT_PER_PAGE),
            **params,
        }
        r = requests.get(BASE, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def search(
        self,
        *,
        state: str | None = None,
        city: str | None = None,
        zip_radius: Tuple[str, str] | None = None,  # (zip, '25mi')
        cip4: int | None = None,
        title_contains: str | None = None,
        page: int = 0,
    ) -> Dict[str, Any]:
        """Search institutions. If CIP code is provided, we filter server‑side; otherwise we pull and filter
        client‑side by program title contains."
        """
        params = {
            "page": page,
            "fields": ",".join(FIELDS),
        }
        if state:
            params["school.state"] = state.upper()
        if city:
            params["school.city"] = city
        if zip_radius:
            zip_code, dist = zip_radius
            params["zip"] = zip_code
            params["distance"] = dist  # e.g., '25mi'
        if cip4:
            params["latest.programs.cip_4_digit.code"] = cip4

        data = self._get(params)

        if title_contains:
            t = title_contains.lower()
            filtered = []
            for row in data.get("results", []):
                titles = row.get("latest.programs.cip_4_digit.title", [])
                if isinstance(titles, list):
                    match = any(t in str(x).lower() for x in titles)
                else:
                    match = t in str(titles).lower()
                if match:
                    filtered.append(row)
            data["results"] = filtered
        return data
