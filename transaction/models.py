from django.contrib.auth.models import User
from django.db import models

from authentication_module.models import signupModel
from companyData.models import companyData


class Transaction(models.Model):
    user = models.EmailField(max_length=254)
    company_symbol = models.CharField(max_length=50)
    transaction_type = models.CharField(max_length=10)
    quantity = models.IntegerField()
    price_per_unit = models.DecimalField(max_digits=10, decimal_places=2)
    transaction_date = models.DateTimeField(auto_now_add=True)

    def total_amount(self):
        return self.quantity * self.price_per_unit


class BuyTransaction(models.Model):
    transaction = models.ForeignKey('Transaction', on_delete=models.CASCADE, related_name='buy_transactions')


class SellTransaction(models.Model):
    transaction = models.ForeignKey('Transaction', on_delete=models.CASCADE, related_name='sell_transactions')


class CreditBalanceUpdate(models.Model):
    user = models.ForeignKey('authentication_module.signupModel', on_delete=models.CASCADE)
    previous_balance = models.DecimalField(max_digits=15, decimal_places=2)
    updated_balance = models.DecimalField(max_digits=15, decimal_places=2)
    update_time = models.DateTimeField(auto_now_add=True)
