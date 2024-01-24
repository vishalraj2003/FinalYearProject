from django.db import models


# Create your models here.
class companyData(models.Model):
    # Existing fields

    companyName = models.CharField(max_length=60)
    symbol = models.CharField(max_length=10, unique=True)
    # last_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

    # New fields based
    one_year_target_est = models.CharField(max_length=10, null=True, blank=True)
    fifty_two_week_range = models.CharField(max_length=50, null=True, blank=True)
    ask = models.CharField(max_length=30, null=True, blank=True)
    avg_volume = models.CharField(max_length=60, null=True, blank=True)
    beta_5y_monthly = models.CharField(max_length=6, null=True, blank=True)
    bid = models.CharField(max_length=30, null=True, blank=True)
    days_range = models.CharField(max_length=50, null=True, blank=True)
    eps_ttm = models.CharField(max_length=10, null=True, blank=True)
    earnings_date = models.CharField(max_length=50, null=True, blank=True)
    ex_dividend_date = models.CharField(max_length=60, null=True, blank=True)
    forward_dividend_yield = models.CharField(max_length=50, null=True, blank=True)
    market_cap = models.CharField(max_length=50, null=True, blank=True)
    open_price = models.CharField(max_length=10, null=True, blank=True)
    pe_ratio_ttm = models.CharField(max_length=10, null=True, blank=True)
    previous_close = models.CharField(max_length=10, null=True, blank=True)
    quote_price = models.CharField(max_length=10, null=True, blank=True)
    volume = models.CharField(max_length=60, null=True, blank=True)
    description = models.TextField(null=True, blank=True)