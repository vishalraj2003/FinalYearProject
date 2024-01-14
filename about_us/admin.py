from django.contrib import admin
from about_us.models import about_us_model


class AboutUsAdmin(admin.ModelAdmin):
    list_display = ('name', 'title', 'description','image')


admin.site.register(about_us_model, AboutUsAdmin)
# Register your models here.
