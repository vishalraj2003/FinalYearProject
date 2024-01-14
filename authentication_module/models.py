from django.db import models


class signupModel(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=30)
    profile_pic = models.ImageField(upload_to='profile_pics/', max_length=200, null=True, blank=True,
                                    default='profile_pics/profile.png')
# Create your models here.
