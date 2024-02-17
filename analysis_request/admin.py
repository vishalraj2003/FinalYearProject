from django.contrib import admin

# Register your models here.
from analysis_request.models import Monitor


class Monitor_admin(admin.ModelAdmin):
    list_display = ('user','continent', 'country', 'city', 'capital', 'datetime', 'ip')


admin.site.register(Monitor, Monitor_admin)