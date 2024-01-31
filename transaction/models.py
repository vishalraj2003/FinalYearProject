from django.contrib.auth.models import User
from django.db import models

from authentication_module.models import signupModel
from companyData.models import companyData


class Transaction(models.Model):
    user = models.EmailField(max_length=254)
    company_symbol = models.CharField(max_length=10)
    transaction_type = models.CharField(max_length=10)
    quantity = models.IntegerField()
    price_per_unit = models.DecimalField(max_digits=10, decimal_places=2)
    transaction_date = models.DateTimeField(auto_now_add=True)


    def total_amount(self):
        return self.quantity * self.price_per_unit

