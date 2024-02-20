from django.contrib import admin


from transaction.models import Transaction, BuyTransaction, SellTransaction, CreditBalanceUpdate



class transactionAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'company_symbol', 'transaction_type', 'quantity', 'price_per_unit', 'transaction_date'
    )


class transaction_buy_Admin(admin.ModelAdmin):
    list_display = ('get_user', 'get_company_symbol', 'get_quantity', 'get_price_per_unit', 'get_transaction_date')

    list_display = ('get_user', 'get_company_symbol', 'get_quantity', 'get_price_per_unit', 'get_transaction_date')

    def get_user(self, obj):
        return obj.transaction.user

    get_user.admin_order_field = 'transaction__user'
    get_user.short_description = 'User'

    def get_company_symbol(self, obj):
        return obj.transaction.company_symbol

    get_company_symbol.admin_order_field = 'transaction__company_symbol'
    get_company_symbol.short_description = 'Company Symbol'

    def get_quantity(self, obj):
        return obj.transaction.quantity

    get_quantity.admin_order_field = 'transaction__quantity'
    get_quantity.short_description = 'Quantity'

    def get_price_per_unit(self, obj):
        return obj.transaction.price_per_unit

    get_price_per_unit.admin_order_field = 'transaction__price_per_unit'
    get_price_per_unit.short_description = 'Price Per Unit'

    def get_transaction_date(self, obj):
        return obj.transaction.transaction_date

    get_transaction_date.admin_order_field = 'transaction__transaction_date'
    get_transaction_date.short_description = 'Transaction Date'


class transaction_sell_Admin(admin.ModelAdmin):
    list_display = ('get_user', 'get_company_symbol', 'get_transaction_type', 'get_quantity', 'get_price_per_unit',
                    'get_transaction_date')

    def get_user(self, obj):
        return obj.transaction.user

    get_user.admin_order_field = 'transaction__user'
    get_user.short_description = 'User'

    def get_company_symbol(self, obj):
        return obj.transaction.company_symbol

    get_company_symbol.admin_order_field = 'transaction__company_symbol'
    get_company_symbol.short_description = 'Company Symbol'

    def get_transaction_type(self, obj):
        return obj.transaction.transaction_type

    get_transaction_type.admin_order_field = 'transaction__transaction_type'
    get_transaction_type.short_description = 'Transaction Type'

    def get_quantity(self, obj):
        return obj.transaction.quantity

    get_quantity.admin_order_field = 'transaction__quantity'
    get_quantity.short_description = 'Quantity'

    def get_price_per_unit(self, obj):
        return obj.transaction.price_per_unit

    get_price_per_unit.admin_order_field = 'transaction__price_per_unit'
    get_price_per_unit.short_description = 'Price Per Unit'

    def get_transaction_date(self, obj):
        return obj.transaction.transaction_date

    get_transaction_date.admin_order_field = 'transaction__transaction_date'
    get_transaction_date.short_description = 'Transaction Date'


class transaction_credit_Admin(admin.ModelAdmin):
    list_display = ('user', 'previous_balance', 'updated_balance', 'update_time')


admin.site.register(Transaction, transactionAdmin)
admin.site.register(BuyTransaction, transaction_buy_Admin)
admin.site.register(SellTransaction, transaction_sell_Admin)
admin.site.register(CreditBalanceUpdate, transaction_credit_Admin)

