from django.db import models


# Create your models here.
class UpdateTracker(models.Model):
    last_run_date = models.DateField(unique=True)
