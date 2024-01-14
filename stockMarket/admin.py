from django.contrib import admin
from authentication_module.models import signupModel


class authentication_admin(admin.ModelAdmin):
    list_display = ('first_name', 'last_name', 'email', 'password','profile_pic')


admin.site.register(signupModel, authentication_admin)
# Register your models here.
