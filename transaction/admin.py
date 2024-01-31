from django.contrib import admin

from transaction.models import Transaction


class transactionAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'company_symbol', 'transaction_type', 'quantity', 'price_per_unit', 'transaction_date'
    )

admin.site.register(Transaction, transactionAdmin)
# Register your models here.
