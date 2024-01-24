from django.contrib import admin
from companyData.models import companyData


# Register your models here.
class companyDataAdmin(admin.ModelAdmin):
    list_display = (
        'companyName', 'symbol', 'one_year_target_est', 'fifty_two_week_range',
        'ask', 'avg_volume', 'beta_5y_monthly', 'bid', 'days_range',
        'eps_ttm', 'earnings_date', 'ex_dividend_date', 'forward_dividend_yield',
        'market_cap', 'open_price', 'pe_ratio_ttm', 'previous_close',
        'quote_price', 'volume', 'description'
    )


admin.site.register(companyData, companyDataAdmin)
