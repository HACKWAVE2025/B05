# tax_calculator.py - Tax calculations by country (CORRECTED CAPITAL GAINS TAX)

from typing import Dict, List


class TaxCalculator:
    """Calculate CAPITAL GAINS taxes by country"""

    TAX_RATES = {
        # Capital Gains Tax Rates (STOCK MARKET) - 2024-2025
        "India": {
            "capital_gains_tax": 0.20,  # 20% LTCG (Long-term capital gains)
            "currency": "INR",
            "name": "India (20% Capital Gains Tax)"
        },
        "USA": {
            "capital_gains_tax": 0.15,  # 15% (long-term capital gains for most)
            "currency": "USD",
            "name": "USA (15% Capital Gains Tax)"
        },
        "UK": {
            "capital_gains_tax": 0.20,  # 20% CGT (higher rate taxpayer)
            "currency": "GBP",
            "name": "UK (20% Capital Gains Tax)"
        },
        "Singapore": {
            "capital_gains_tax": 0.0,  # NO capital gains tax
            "currency": "SGD",
            "name": "Singapore (0% - No Capital Gains Tax)"
        },
        "Dubai": {
            "capital_gains_tax": 0.0,  # NO capital gains tax
            "currency": "AED",
            "name": "Dubai/UAE (0% - No Capital Gains Tax)"
        },
        "Germany": {
            "capital_gains_tax": 0.26,  # 26% (Solidaritätszuschlag included)
            "currency": "EUR",
            "name": "Germany (26% Capital Gains Tax)"
        },
        "Canada": {
            "capital_gains_tax": 0.27,  # 50% of gains included in taxable income (~27% effective)
            "currency": "CAD",
            "name": "Canada (27% Effective Capital Gains Tax)"
        },
        "Australia": {
            "capital_gains_tax": 0.23,  # 50% discount on capital gains (~23% effective)
            "currency": "AUD",
            "name": "Australia (23% Effective Capital Gains Tax)"
        },
        "Japan": {
            "capital_gains_tax": 0.20,  # 20% flat tax on stock gains
            "currency": "JPY",
            "name": "Japan (20% Capital Gains Tax)"
        },
        "France": {
            "capital_gains_tax": 0.36,  # ~36% (income tax + social contributions)
            "currency": "EUR",
            "name": "France (36% Capital Gains Tax)"
        }
    }

    def get_tax_rate(self, country: str) -> float:
        """Get capital gains tax rate for country"""
        if country in self.TAX_RATES:
            return self.TAX_RATES[country].get("capital_gains_tax", 0.0)
        return 0.0

    def calculate_after_tax_return(self, gross_return: float, country: str) -> Dict:
        """Calculate after-tax returns"""
        if country not in self.TAX_RATES:
            return {"error": f"Country {country} not found"}

        tax_rate = self.get_tax_rate(country)
        tax_amount = gross_return * tax_rate
        net_return = gross_return - tax_amount

        return {
            "gross_return": round(gross_return, 2),
            "tax_rate": round(tax_rate * 100, 1),
            "tax_amount": round(tax_amount, 2),
            "net_return": round(net_return, 2),
            "country": country,
            "tax_name": self.TAX_RATES[country]["name"]
        }

    def get_all_countries(self) -> List[str]:
        """Get all supported countries"""
        return list(self.TAX_RATES.keys())

    def get_country_info(self, country: str) -> Dict:
        """Get country tax info"""
        if country in self.TAX_RATES:
            info = self.TAX_RATES[country].copy()
            info["capital_gains_tax"] = info.get("capital_gains_tax", 0.0)
            return info
        return {"capital_gains_tax": 0.0}

    def get_tax_comparison(self, gross_return: float) -> Dict:
        """Compare after-tax returns across all countries"""
        comparison = {}
        for country in self.get_all_countries():
            result = self.calculate_after_tax_return(gross_return, country)
            comparison[country] = result
        return comparison


# Create global instance
tax_calculator = TaxCalculator()
