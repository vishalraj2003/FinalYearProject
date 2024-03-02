from django.db import models

# Create your models here.
from django.db import models

class credit_balance_update(models.Model):
    email = models.EmailField()
    previous_balance = models.DecimalField(max_digits=10, decimal_places=2)
    transaction_type = models.CharField(max_length=10, null=True)
    current_balance = models.DecimalField(max_digits=10, decimal_places=2)
    transaction_date = models.DateTimeField(auto_now_add=True)

    def total_amount(self):
        return self.current_balance - self.previous_balance
