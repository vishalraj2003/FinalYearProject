from django.db import models


class about_us_model(models.Model):
    name = models.CharField(max_length=50)
    title = models.CharField(max_length=100)
    description = models.TextField()
    image = models.ImageField(upload_to='about-us/', max_length=100, null=True, blank=True,
                              default='http://placehold.it/200x200')

# Create your models here.
