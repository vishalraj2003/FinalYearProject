from django.contrib import admin

# Register your models here.
from django.contrib import admin
from credit_balance_update.models import credit_balance_update


# Register your models here.
class credit_balance_update_admin(admin.ModelAdmin):
    list_display = (
        'email', 'previous_balance','transaction_type','current_balance', 'transaction_date',
    )


admin.site.register(credit_balance_update, credit_balance_update_admin)
