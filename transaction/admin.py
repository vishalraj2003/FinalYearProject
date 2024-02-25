from django.contrib import admin


from transaction.models import Transaction, BuyTransaction, SellTransaction, CreditBalanceUpdate



class transactionAdmin(admin.ModelAdmin):
    list_display = (
        'id','user', 'company_symbol', 'transaction_type', 'quantity', 'buy_price_per_unit','sell_price_per_unit', 'transaction_date'
    )


class transaction_buy_Admin(admin.ModelAdmin):
    list_display = (
        'id', 'user', 'company_symbol', 'transaction_type', 'quantity', 'buy_price_per_unit', 'sell_price_per_unit',
        'transaction_date'
    )


class transaction_sell_Admin(admin.ModelAdmin):
    list_display = (
        'id', 'user', 'company_symbol', 'transaction_type', 'quantity', 'buy_price_per_unit', 'sell_price_per_unit',
        'sell_profit_loss','transaction_date'
    )


class transaction_credit_Admin(admin.ModelAdmin):
    list_display = ('user', 'previous_balance', 'updated_balance', 'update_time')


admin.site.register(Transaction, transactionAdmin)
admin.site.register(BuyTransaction, transaction_buy_Admin)
admin.site.register(SellTransaction, transaction_sell_Admin)
admin.site.register(CreditBalanceUpdate, transaction_credit_Admin)

