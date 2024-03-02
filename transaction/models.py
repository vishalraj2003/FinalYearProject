from django.contrib.auth.models import User
from django.db import models

from authentication_module.models import signupModel
from companyData.models import companyData


class Transaction(models.Model):
    user = models.EmailField(max_length=254,null=True)
    company_symbol = models.CharField(max_length=50,null=True)
    transaction_type = models.CharField(max_length=10,null=True)
    quantity = models.IntegerField(null=True)
    buy_price_per_unit = models.DecimalField(max_digits=10, decimal_places=2,null=True)
    sell_price_per_unit = models.DecimalField(max_digits=10, decimal_places=2,null=True)
    transaction_date = models.DateTimeField(auto_now_add=True,null=True)

    def total_amount(self):
        return self.quantity * self.price_per_unit


class BuyTransaction(models.Model):
    user = models.EmailField(max_length=254,null=True)
    company_symbol = models.CharField(max_length=50,null=True)
    transaction_type = models.CharField(max_length=10,null=True)
    quantity = models.IntegerField(null=True)
    buy_price_per_unit = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    sell_price_per_unit = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    transaction_date = models.DateTimeField(auto_now_add=True,null=True)

    def total_amount(self):
        return self.quantity * self.buy_price_per_unit


class SellTransaction(models.Model):
    user = models.EmailField(max_length=254,null=True)
    company_symbol = models.CharField(max_length=50,null=True)
    transaction_type = models.CharField(max_length=10,null=True)
    quantity = models.IntegerField(null=True)
    buy_price_per_unit = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    sell_price_per_unit = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    sell_profit_loss = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    transaction_date = models.DateTimeField(auto_now_add=True,null=True)


    def total_amount(self):
        return self.quantity * self.sell_price_per_unit


class CreditBalanceUpdate(models.Model):
    user = models.EmailField(max_length=254,null=True)
    previous_balance = models.DecimalField(max_digits=15, decimal_places=2)
    updated_balance = models.DecimalField(max_digits=15, decimal_places=2)
    update_time = models.DateTimeField(auto_now_add=True)
